import tensorflow as tf

a = tf.constant(4, name = "a")
b = tf.constant(2, name = "b")
c = tf.multiply(a,b, name ="c")
d = tf.add(a,b, name = "d")
e = tf.add(c,d, name = "e")

with tf.Session() as sess: 
    print(sess.run(e))
    wirter = tf.summary.FileWriter('./my_graph', sess.graph)