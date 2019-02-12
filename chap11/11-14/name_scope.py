import tensorflow as tf

with tf.name_scope('hidden') as scope:
    a = tf.constant(5, name='alpha')
    print(a.name)
    weights = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
    print(weights.name)
    bias = tf.Variable(tf.zeros([1]), name='biases')
    print(bias.name)

with tf.name_scope('conv1') as scope:
    weights = tf.Variable([1.0, 2.0], name='weights')
    print(weights.name)
    bias = tf.Variable([0.3], name='biases')
    print(bias.name)
    
sess = tf.Session()
writer = tf.summary.FileWriter('./my_graph/2', sess.graph)
