import tensorflow as tf
from tensorflow import keras 
from keras import layers

def smart_teleport(edge_index, alpha=0.05, iter=1000):
    edge_index = tf.squeeze(edge_index)
    T = tf.math.bincount(edge_index[:,0])
    n_nodes = T.shape[0]
    diag_indices = tf.concat([tf.expand_dims(tf.range(n_nodes), axis=-1), tf.expand_dims(tf.range(n_nodes), axis=-1)], axis=-1)
    diagonal = tf.sparse.SparseTensor(tf.cast(diag_indices, tf.int64), -1*tf.ones([n_nodes,], dtype=tf.float32), dense_shape=[n_nodes, n_nodes])
    A = tf.sparse.reorder(tf.sparse.SparseTensor(tf.cast(edge_index, tf.int64), tf.ones([edge_index.shape[0],], dtype=tf.float32), [n_nodes, n_nodes]))
    A = tf.sparse.add(A, diagonal)

    T = tf.sparse.from_dense(tf.transpose(tf.math.multiply(tf.sparse.to_dense(tf.sparse.transpose(A)), tf.broadcast_to(tf.transpose(tf.math.reciprocal_no_nan(tf.reduce_sum(tf.sparse.to_dense(A), axis=1, keepdims=True))), shape=A.shape))))

    e_v = tf.reduce_sum(tf.sparse.to_dense(A), axis=0) / tf.reduce_sum(tf.sparse.to_dense(A))

    p = e_v

    for _ in range(iter):
        p = tf.math.scalar_mul(alpha, e_v) + tf.math.scalar_mul((1.0 - alpha), tf.squeeze(tf.matmul(tf.expand_dims(p, axis=0), tf.sparse.to_dense(T))))


    F = tf.sparse.from_dense(tf.math.scalar_mul(alpha, tf.sparse.to_dense(A) / tf.reduce_sum(tf.sparse.to_dense(A))) + tf.transpose(tf.math.scalar_mul((1.0 - alpha), tf.broadcast_to(p, T.dense_shape)*tf.transpose(tf.sparse.to_dense(T)))))

    return F, p




class NeuromapPooling(layers.Layer):
    def __init__(self, edge_index, n_clusters, alpha: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.F, self.p = smart_teleport(edge_index)
        self.p_log_p = tf.reduce_sum(tf.math.multiply_no_nan(log2(self.p), self.p))
        self.n_clusters = n_clusters
    def call(self, inputs):
        eps = 1e-8
        features, S = inputs

        S = S + eps
        identity = tf.eye(self.n_clusters, self.n_clusters)
        C = tf.matmul(tf.matmul(S, tf.sparse.to_dense(self.F), transpose_a=True), S)
        q = 1.0 - tf.linalg.trace(C)
        q_m = tf.reduce_sum(C, axis=1) - tf.linalg.diag_part(C)
        m_exit = tf.reduce_sum(C, axis=0) - tf.linalg.diag_part(C)
        p_m = q_m + tf.reduce_sum(C, axis=0)

        codelength = tf.reduce_sum(tf.math.multiply_no_nan(log2(q), q)) - tf.reduce_sum(tf.math.multiply_no_nan(log2(q_m), q_m)) - tf.reduce_sum(tf.math.multiply_no_nan(log2(m_exit), m_exit)) - self.p_log_p \
        + tf.reduce_sum(tf.math.multiply_no_nan(log2(p_m), p_m))

        # S = S + eps
        # identity = tf.eye(self.n_clusters, self.n_clusters)
        # out_adj = tf.matmul(tf.matmul(S, tf.sparse.to_dense(self.F), transpose_a=True), S)
        # diag = tf.math.multiply(identity, tf.linalg.diag_part(out_adj))

        # e1 = tf.reduce_sum(out_adj) - tf.linalg.trace(out_adj)
        # e2 = tf.reduce_sum(tf.math.subtract(out_adj, diag), axis=-1)
        # e3 = self.p
        # e4 = tf.reduce_sum(out_adj, axis=-1) + tf.reduce_sum(tf.math.subtract(tf.transpose(out_adj), diag), -1)

        # e1 = tf.reduce_sum(tf.math.multiply_no_nan(log2(e1), e1))
        # e2 = tf.reduce_sum(tf.math.multiply_no_nan(log2(e2), e2))
        # e3 = tf.reduce_sum(tf.math.multiply_no_nan(log2(e3), e3))
        # e4 = tf.reduce_sum(tf.math.multiply_no_nan(log2(e4), e4))

        # codelength = e1 - 2*e2 - e3 + e4
        out = tf.matmul(S, features, transpose_a=True)
        return out, S, codelength

def log2(x: tf.Tensor):
    return tf.math.divide(tf.math.log(x), tf.math.log(tf.constant(2, dtype=x.dtype)))