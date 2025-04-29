from config import load_config
from data.data_loader import SnapLoader
from src.custom_layers import RFPEncoding
from src.pool import smart_teleport
from src.custom_layers import GCNRFPEncode
import tensorflow as tf
from src.model import GCNCommunityModel

config = load_config("config")
print(config["dataset"]["name"])
rfp_config = config["positional_encoding"]
community_config = config["community_gnn"]

_edge_index, _node_features = SnapLoader(config["dataset"]["name"],"email-Eu-core.txt.gz","test", True, True, feature_config=None).graph_tensor()

# model = GCNCommunityModel(_edge_index, community_config, rfp_config)
# result = model(_node_features)
# print(result[0])
# print(result[1])
# print(result[2])

F, p = smart_teleport(_edge_index)
print(tf.reduce_any(tf.math.is_nan(tf.sparse.to_dense(F))))