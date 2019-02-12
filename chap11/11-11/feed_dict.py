import tensorflow as tf

a = tf.add(1,2)
b = tf.multiply(a,2)
session = tf.Session()
v1 = session.run(b)
print (v1)

replace_dict = {a:20}
session.run(b, feed_dict = replace_dict)
