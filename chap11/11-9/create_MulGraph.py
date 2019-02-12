import tensorflow as tf
import numpy as np

g1 = tf.Graph()
g2 = tf.Graph()
with g1.as_default():
    a = tf.constant(123)
    print(a.graph)
    print(tf.get_default_graph())

with g2.as_default():
    b = tf.multiply(2,3)
    print(b.graph)
    print(tf.get_default_graph())