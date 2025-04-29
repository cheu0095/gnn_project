import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow import keras 
from keras import layers
from config import load_config
from data.data_loader import SnapLoader
from src.custom_layers import RFPEncoding
from src.pool import smart_teleport
from src.custom_layers import GCNRFPEncode
from src.model import GCNCommunityModel
from sklearn.metrics import adjusted_mutual_info_score

config = load_config("config")
print(config["dataset"]["name"])
rfp_config = config["positional_encoding"]
community_config = config["community_gnn"]

dataset = SnapLoader(config["dataset"]["name"],"email-Eu-core.txt.gz","email-Eu-core-department-labels.txt", True, True, feature_config=None)

_edge_index, _node_features = dataset.graph_tensor()
_labels = dataset._load_labels()




def GNN_Community(edge_index, feature_shape):
    node_feature = layers.Input(feature_shape, batch_size=1, dtype=tf.float32)
    community_gnn = GCNCommunityModel(edge_index, community_config, rfp_config)
    module_features, module_adj, loss = community_gnn(node_feature)

    
    model = keras.Model(inputs=node_feature, outputs=module_adj,)
    model.add_loss(loss)
    return model

gnn_com = GNN_Community(_edge_index, _node_features.shape[1:])
gnn_com.compile(loss=None, optimizer=tf.keras.optimizers.Adam(learning_rate=0.1, decay=0.0001))

callback = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=200, min_lr=0.0001)
history = gnn_com.fit(_node_features, epochs=15000, callbacks=callback)
cluster = gnn_com.predict(_node_features, batch_size=1)

_labels_pred = tf.cast(tf.constant(tf.math.argmax(cluster, axis=-1)), tf.int8).numpy()

print(_labels_pred)
np.savetxt("results\\clusters.txt", _labels_pred)

print(adjusted_mutual_info_score(_labels[:,1], _labels_pred))
