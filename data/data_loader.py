import gzip
import tensorflow as tf
from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
from typing import Optional, Tuple, Any


class GraphLoader(ABC):
    @abstractmethod
    def __init__(self, name: str,  is_directed: bool, has_classes: bool, feature_config: Optional[dict] = None):
        self.name = name
        self.feature_config = feature_config
        self.is_directed = is_directed
        self.has_classes = has_classes

    @abstractmethod
    def load_adjacency(self) -> tf.Tensor:
        ...

    @abstractmethod
    def load_features(self) -> tf.Tensor:
        ...

    @abstractmethod
    def _to_networkx(self) -> nx.Graph:
        ...

    def graph_tensor(self) -> Tuple[tf.Tensor]:
        G = self._to_networkx()
        return tuple((self.load_adjacency(G), self.load_features(G)))
    
class SnapLoader(GraphLoader):
    def __init__(self, name: str, graph_file: str, label_file: str, is_directed: bool, has_classes: bool, feature_config: Optional[dict] = None):
        super().__init__(name, is_directed, has_classes, feature_config)
        self.graph_file = graph_file
        self.label_file = label_file

    def _to_networkx(self):
        with gzip.open("data\\inputs\\" + self.name + "\\" + self.graph_file, "rb") as file:
            constructor = nx.DiGraph if self.is_directed else nx.Graph
            G = nx.read_edgelist(file, create_using=constructor, comments="#", data=False, nodetype=int)
        #Add self loops
        for i in G.nodes:
            G.add_edge(i,i)

        #Reindex Node labels
        G = nx.convert_node_labels_to_integers(G, first_label=0)
        return G
    
    def load_adjacency(self, G) -> tf.Tensor:
        return tf.expand_dims(tf.convert_to_tensor(list(G.edges(data=False)), dtype=tf.int32), axis=0)
    
    def load_features(self, G) -> tf.Tensor:
        if self.feature_config is None:
            return tf.expand_dims(tf.expand_dims(tf.convert_to_tensor([deg for (_, deg) in G.degree()], dtype=tf.float32), axis=0), axis=-1)
        else:
            raise NotImplementedError

    def _load_labels(self):
        with open("data\\inputs\\" + self.name + "\\" + self.label_file, "rb") as f:
            array = tf.cast(tf.constant(np.loadtxt(f)), tf.int8)
        return array


class NodeFeatureEncoder:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.featuresMap = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.featuresMap[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)
    def encode(self, graph: nx.Graph):
        graphSize = graph.number_of_nodes()
        output = np.zeros((graphSize, self.dim + 1))
        graphNodes = graph.nodes
        graphDegree = graph.degree
        for featureName, featureMap in self.featuresMap.items():
            rowNum = 0
            for node in graphNodes:
                value = graphNodes[node][featureName]
                if value not in featureMap:
                    continue
                output[rowNum][featureMap[value]]=1.0
                output[rowNum][self.dim]=graphDegree[node]
                rowNum +=1
        return output

    
