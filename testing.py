import tensorflow as tf

id = tf.eye(2,2)
diag = id*tf.constant([2.0,3.0])
print(diag)