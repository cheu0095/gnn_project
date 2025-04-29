import tensorflow as tf
from tensorflow import keras 
from keras import layers

class GCNEncoder(layers.Layer):
    def __init__(self, hiddenDim, dropout_prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.hiddenDim = hiddenDim
        self.dropout = layers.Dropout(dropout_prob)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        
    def build(self, input_shape):
        self.featureDim = input_shape[1][-1]
        self.kernel = self.add_weight(shape=(self.featureDim, self.hiddenDim), initializer="glorot_uniform", dtype=tf.float32)
        self.bias = self.add_weight(shape=(self.hiddenDim), initializer="zeros", dtype=tf.float32)
        self.built = True
        
    def call(self, inputs, training=False):
        edge_pairs, node_features = inputs
        transformed_features = self.dropout(tf.matmul(node_features, self.kernel) + self.bias, training)
        neighbour_features = tf.gather(transformed_features, edge_pairs[:,1])
        source_features = tf.gather(transformed_features, edge_pairs[:,0])
        ones_degree = tf.ones((tf.shape(edge_pairs)[0]))
        node_degree_ = tf.math.unsorted_segment_sum(ones_degree, edge_pairs[:,0], tf.shape(node_features)[0])
        node_degree = tf.math.pow(node_degree_, -0.5)
        
        start_degree = tf.gather(node_degree, edge_pairs[:,0])
        source_degree = tf.broadcast_to(tf.expand_dims(tf.gather(node_degree_, edge_pairs[:,0]), axis=-1), [start_degree.shape[0], self.hiddenDim])
        end_degree = tf.gather(node_degree, edge_pairs[:,1])
        neighbour_features = tf.einsum('j,jk->jk', start_degree, neighbour_features)
        neighbour_features = tf.einsum('j,jk->jk', end_degree, neighbour_features)
        
        aggregated_features = 0.5 * tf.math.unsorted_segment_sum(neighbour_features, edge_pairs[:,0], tf.shape(node_features)[0])
        aggregated_features = aggregated_features + (0.5 * tf.math.unsorted_segment_mean(tf.math.multiply(source_degree, source_features), edge_pairs[:,0], tf.shape(node_features)[0]))
        
        return self.batch_norm(aggregated_features)

class GCNDecoder(layers.Layer):
    def __init__(self, hiddenDim, dropout_prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.hiddenDim = hiddenDim
        self.dropout = layers.Dropout(dropout_prob)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        
    def build(self, input_shape):
        self.featureDim = input_shape[1][-1]
        self.kernel = self.add_weight(shape=(self.featureDim, self.hiddenDim), initializer="glorot_uniform", dtype=tf.float32)
        self.bias = self.add_weight(shape=(self.hiddenDim), initializer="zeros", dtype=tf.float32)
        self.built = True
        
    def call(self, inputs, training=False):
        edge_pairs, node_features = inputs
        transformed_features = self.dropout(tf.matmul(node_features, self.kernel) + self.bias, training)
        neighbour_features = tf.gather(transformed_features, edge_pairs[:,1])
        source_features = tf.gather(transformed_features, edge_pairs[:,0])
        ones_degree = tf.ones((tf.shape(edge_pairs)[0]))
        node_degree_ = tf.math.unsorted_segment_sum(ones_degree, edge_pairs[:,0], tf.shape(node_features)[0])
        node_degree = tf.math.pow(node_degree_, -0.5)
        
        start_degree = tf.gather(node_degree, edge_pairs[:,0])
        source_degree = tf.broadcast_to(tf.expand_dims(tf.gather(node_degree_, edge_pairs[:,0]), axis=-1), [start_degree.shape[0], self.hiddenDim])
        end_degree = tf.gather(node_degree, edge_pairs[:,1])
        neighbour_features = tf.einsum('j,jk->jk', start_degree, neighbour_features)
        neighbour_features = tf.einsum('j,jk->jk', end_degree, neighbour_features)
        
        aggregated_features = 0.5 * tf.math.unsorted_segment_sum(neighbour_features, edge_pairs[:,0], tf.shape(node_features)[0])
        aggregated_features = (0.5 * tf.math.unsorted_segment_mean(tf.math.multiply(source_degree, source_features), edge_pairs[:,0], tf.shape(node_features)[0])) - aggregated_features
        
        return self.batch_norm(aggregated_features)

class RFPEncoding(layers.Layer):
    def __init__(self, edge_index, initial_dim: int, node_dim: int, batch_dim: int, layer_dim: int, randseed: int, **kwargs):
        super().__init__(trainable=True, **kwargs)
        self.propagater = RFPConv(edge_index)
        self.node_dim = node_dim
        self.initial_dim = initial_dim
        self.layer_dim = layer_dim
        self.batch_dim = batch_dim
        self.randseed = randseed
        
    def call(self, run):
        if run:
            trajectory = [tf.random.normal([self.batch_dim, self.node_dim, self.initial_dim], 0, 1, tf.float32, seed=(self.randseed))]
            for i in range(self.layer_dim):
                trajectory.append(tf.map_fn(fn=self.propagater, elems=trajectory[i], parallel_iterations=self.batch_dim))
            return tf.concat(trajectory, axis=-1)
    
class RFPConv(layers.Layer):
    def __init__(self, edge_index):
        super().__init__(trainable=True)
        self.edge_index = tf.squeeze(edge_index)
        
    def call(self, node_features):
        edge_pairs = self.edge_index
        transformed_features = node_features
        neighbour_features = tf.gather(transformed_features, edge_pairs[:,1])
        source_features = tf.gather(transformed_features, edge_pairs[:,0])
        ones_degree = tf.ones((tf.shape(edge_pairs)[0]))
        node_degree_ = tf.math.unsorted_segment_sum(ones_degree, edge_pairs[:,0], tf.shape(node_features)[0])
        node_degree = tf.math.pow(node_degree_, -0.5)
        
        start_degree = tf.gather(node_degree, edge_pairs[:,0])
        source_degree = tf.broadcast_to(tf.expand_dims(tf.gather(node_degree_, edge_pairs[:,0]), axis=-1), [start_degree.shape[0], node_features.shape[-1]])
        end_degree = tf.gather(node_degree, edge_pairs[:,1])
        neighbour_features = tf.einsum('j,jk->jk', start_degree, neighbour_features)
        neighbour_features = tf.einsum('j,jk->jk', end_degree, neighbour_features)
        
        aggregated_features = 0.5 * tf.math.unsorted_segment_sum(neighbour_features, edge_pairs[:,0], tf.shape(node_features)[0])
        aggregated_features = aggregated_features + (0.5 * tf.math.unsorted_segment_mean(tf.math.multiply(source_degree, source_features), edge_pairs[:,0], tf.shape(node_features)[0]))
        
        return tf.linalg.qr(aggregated_features, full_matrices=True)[0]
    
class GCNRFPEncode(layers.Layer):
    def __init__(self, hiddenDim, edge_index, dropout_prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.hiddenDim = hiddenDim
        self.edge_index = edge_index
        self.dropout = layers.Dropout(dropout_prob)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        
    def build(self, input_shape):
        self.featureDim = input_shape[-1]
        self.kernel = self.add_weight(shape=(self.featureDim, self.hiddenDim), initializer="glorot_uniform", dtype=tf.float32)
        self.bias = self.add_weight(shape=(self.hiddenDim), initializer="zeros", dtype=tf.float32)
        self.built = False
        
    def call(self, node_features, training=False):
        edge_pairs = tf.squeeze(self.edge_index)
        transformed_features = self.dropout(tf.matmul(node_features, self.kernel) + self.bias, training)
        neighbour_features = tf.gather(transformed_features, edge_pairs[:,1])
        source_features = tf.gather(transformed_features, edge_pairs[:,0])
        ones_degree = tf.ones((tf.shape(edge_pairs)[0]))
        node_degree_ = tf.math.unsorted_segment_sum(ones_degree, edge_pairs[:,0], tf.shape(node_features)[0])
        node_degree = tf.math.pow(node_degree_, -0.5)
        
        start_degree = tf.gather(node_degree, edge_pairs[:,0])
        source_degree = tf.broadcast_to(tf.expand_dims(tf.gather(node_degree_, edge_pairs[:,0]), axis=-1), [start_degree.shape[0], self.hiddenDim])
        end_degree = tf.gather(node_degree, edge_pairs[:,1])
        neighbour_features = tf.einsum('j,jk->jk', start_degree, neighbour_features)
        neighbour_features = tf.einsum('j,jk->jk', end_degree, neighbour_features)
        
        aggregated_features = 0.5 * tf.math.unsorted_segment_sum(neighbour_features, edge_pairs[:,0], tf.shape(node_features)[0])
        aggregated_features = aggregated_features + (0.5 * tf.math.unsorted_segment_mean(tf.math.multiply(source_degree, source_features), edge_pairs[:,0], tf.shape(node_features)[0]))
        
        return self.batch_norm(aggregated_features)