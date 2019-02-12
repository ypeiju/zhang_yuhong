import tensorflow as tf
import numpy as np

A = list([1,2,3,])
B = np.array([4, 5, 6], dtype=np.int32)
C = tf.convert_to_tensor(A)
D = tf.convert_to_tensor(B)
E = tf.add(A, B)
with tf.Session() as sess:
    print(type(A))
    print(type(B))
    print(type(C))
    print(type(D))
    print(sess.run(E))
