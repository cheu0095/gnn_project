import tensorflow as tf
from tensorflow import keras 
from keras import layers
from src.custom_layers import GCNEncoder, GCNDecoder, RFPEncoding, GCNRFPEncode
from src.pool import NeuromapPooling
from typing import Optional, Any

class GCNCommunityModel(layers.Layer):
    def __init__(self, edge_index, community_config: dict, rfp_config: dict, **kwargs):
        super().__init__(**kwargs)
        self.edge_index = tf.squeeze(edge_index)
        self.encoder_spec = community_config['layer_spec']
        self.num_layers = len(self.encoder_spec)
        self.decoder_spec = [community_config['max_communities']] + self.encoder_spec
        self.decoder_spec.reverse()
        self.decoder_spec.pop(0)
        self.encoder = [GCNEncoder(spec) for spec in self.encoder_spec]
        self.rfp_encoder = [GCNRFPEncode(spec, self.edge_index) for spec in self.encoder_spec]
        self.rfp_encoder2 = [GCNEncoder(spec) for spec in self.encoder_spec]
        self.decoder = [GCNDecoder(spec) for spec in self.decoder_spec]
        self.dropout = layers.Dropout(community_config['dropout'])
        self.rfp_config = rfp_config
        self.rfp_embed = keras.Sequential([keras.layers.Dense(rfp_config['embed_dim'], use_bias=False)])
        self.pool = NeuromapPooling(self.edge_index, community_config['max_communities'])
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        self.nodedim = input_shape[1]
        self.rfp_trajectory = RFPEncoding(self.edge_index, initial_dim=self.rfp_config['initial_dim'], node_dim=self.nodedim, batch_dim=self.rfp_config['num_batches'], layer_dim=self.rfp_config['num_layers'], randseed=self.rfp_config['seed'])(True)
        self.built = True
    
    def call(self, node_features : tf.Tensor, training=False):
        rfp_trajectory = self.batch_norm(self.rfp_embed(tf.concat([self.rfp_trajectory, tf.broadcast_to(node_features, shape=self.rfp_trajectory.shape)], axis=-1)))
        rfp_trajectory_sum = tf.reduce_sum(rfp_trajectory, axis=0, keepdims=False)
        for i in range(0, self.num_layers):
            if i < self.num_layers - 1:
                rfp_trajectory = tf.nn.relu(tf.map_fn(fn=self.rfp_encoder[i], elems=rfp_trajectory, parallel_iterations=self.rfp_config['num_batches']))
                rfp_trajectory_sum = tf.nn.relu(self.rfp_encoder2[i]((self.edge_index, rfp_trajectory_sum)))

            else:
                rfp_trajectory = tf.map_fn(fn=self.rfp_encoder[i], elems=rfp_trajectory, parallel_iterations=self.rfp_config['num_batches'])
                rfp_trajectory_sum = self.rfp_encoder2[i]((self.edge_index, rfp_trajectory_sum))
            rfp_trajectory = rfp_trajectory + rfp_trajectory_sum
        node_features = tf.reduce_sum(rfp_trajectory + rfp_trajectory_sum, axis=0)
        encoded_features = node_features

        for i in range(0, self.num_layers):
            if i < self.num_layers - 1:
                node_features = tf.nn.relu(self.decoder[i]((self.edge_index, node_features)))
            else:
                node_features = tf.nn.softmax(self.decoder[i]((self.edge_index, node_features)), axis=-1)
            
        out, out_adj, map_eq_loss = self.pool((encoded_features, node_features))
        return out, node_features, map_eq_loss