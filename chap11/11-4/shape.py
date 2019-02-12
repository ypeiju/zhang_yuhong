import tensorflow as tf
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
tensor_shape = tf.shape(t)  # [2, 2, 3]
with tf.Session() as sess: 
    print(sess.run(tensor_shape))
