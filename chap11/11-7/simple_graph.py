import tensorflow as tf
import numpy as np
  
a = tf.constant(123)
print(a.graph)
print(tf.get_default_graph())
