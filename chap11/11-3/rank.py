import tensorflow as tf
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
tensor_rank = tf.rank(t)  # [2, 2, 3]
with tf.Session() as sess: 
    print(sess.run(tensor_rank))      #输出结果为3
