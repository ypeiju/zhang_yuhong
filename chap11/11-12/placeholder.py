import tensorflow as tf

a = tf.placeholder(tf.float32, name = "input_1")
b = tf.placeholder(tf.float32, name = "input_2")
output = tf.multiply(a, b, name = "mul_out")

input_dict = {a : 7.0, b : 10.0}

with tf.Session() as session:
    print(session.run(output, feed_dict = input_dict))
