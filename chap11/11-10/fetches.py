import tensorflow as tf
from collections import namedtuple

a = tf.constant([10, 20])
b = tf.constant([1.0, 2.0])
session = tf.Session()

v1 = session.run(a)
print(v1)
v2 = session.run([a, b])
print(v2)

