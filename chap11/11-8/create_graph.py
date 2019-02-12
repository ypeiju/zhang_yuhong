import tensorflow as tf
import numpy as np
  
g = tf.Graph()
with g.as_default():
    a = tf.constant(123)
    print(a.graph)
    print(tf.get_default_graph())
