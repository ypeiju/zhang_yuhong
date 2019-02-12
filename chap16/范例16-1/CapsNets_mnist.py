############################################
###  filename: CapsNet_mnist.py 
###  Function: TensorFlow实现 CapsNet_mnist.py
### Reference：https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb
###     Paper：Dynamic Routing Between Capsules---
###            Authored by Sara Sabour, Nicholas Frosst and Geoffrey E. Hinton (NIPS 2017).
############################################

'''Imports'''
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

'''Input Images'''
X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")

''' 卷积层'''
conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}

conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)

'''Primary Capsule'''
caps1_num  = 32
caps1_dims = 8

conv2_params = {
    "filters": caps1_num * caps1_dims, # 256 convolutional filters
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}

conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

caps1_caps = caps1_num * 6 * 6

caps1_raw = tf.reshape(conv2, [-1, caps1_caps, caps1_dims],
                       name="caps1_raw")

def squash(s, axis = -1, epsilon=1e-7):
    s_sqr_norm = tf.reduce_sum(tf.square(s), axis = axis,
                                 keepdims=True)
    V = s_sqr_norm /(1. + s_sqr_norm)/tf.sqrt(s_sqr_norm + epsilon)
    return V * s

caps1_output = squash(caps1_raw)

'''Digit Capsules'''
caps2_caps  = 10
caps2_dims  = 16

routing_num = 2

init_sigma  = 0.01

W_init = tf.random_normal(
    shape=(1, caps1_caps, caps2_caps, caps2_dims, caps1_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")

batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

# 数组 u
caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_caps, 1, 1],
                             name="caps1_output_tiled")

# 数组 u_hat							 
caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")

'''路由算法'''
raw_weights = tf.zeros([batch_size, caps1_caps, caps2_caps, 1, 1],
                       dtype=tf.float32, name="raw_weights")
b = raw_weights

for i in range(routing_num):
    c = tf.nn.softmax(b, axis = 2) # b == raw weights
    preds = tf.multiply(c, caps2_predicted)
    s = tf.reduce_sum(preds, axis=1, keepdims=True)
    vj = squash(s, axis=-2)

    if i < routing_num - 1:
        vj_tiled = tf.tile(vj, [1, caps1_caps, 1, 1, 1], name= "vj_tiled")

        agreement = tf.matmul(caps2_predicted, vj_tiled, transpose_a=True, name="agreement")

        b += agreement

caps2_output = vj

'''求预测向量长度'''
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)

y_proba = safe_norm(vj, axis = -2, name="y_proba")
y_proba_argmax = tf.argmax(y_proba, axis = 2, name = "y_proba")
y_pred = tf.squeeze(y_proba_argmax, axis = [1,2], name = "y_pred")


'''labels'''
y = tf.placeholder(shape=[None], dtype=tf.int64)

'''Margin loss'''
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

T = tf.one_hot(y, depth=caps2_caps, name="T")

caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm")

present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                           name="present_error")

absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                          name="absent_error")

L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")

margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

'''Mask'''
mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")

reconstruction_targets = tf.cond(mask_with_labels, # condition
                                 lambda: y,        # if True
                                 lambda: y_pred,   # if False
                                 name="reconstruction_targets")


reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=caps2_caps,
                                 name="reconstruction_mask")

reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, caps2_caps, 1, 1],
    name="reconstruction_mask_reshaped")

caps2_output_masked = tf.multiply(
    caps2_output, reconstruction_mask_reshaped,
    name="caps2_output_masked")

'''Decoder'''
n_hidden1 = 512
n_hidden2 = 1024
n_output = 28 * 28
	
decoder_input = tf.reshape(caps2_output_masked,
                           [-1, caps2_caps * caps2_dims],
                           name="decoder_input")

with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")

'''重构损失'''									 
X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_sum(squared_difference,
                                    name="reconstruction_loss")

'''最终损失'''
alpha = 0.0005
loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

'''额外设置'''

'''Accuracy'''
correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

'''用 Adam 优化器'''
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")

'''全局初始化'''
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#############################

'''Training'''
n_epochs = 3
batch_size = 50
restore_checkpoint = True

n_iterations_per_epoch = mnist.train.num_examples // batch_size
n_iterations_validation = mnist.validation.num_examples // batch_size
#print("----",n_iterations_validation)
best_loss_val = np.infty
checkpoint_path = "./my_capsule_network"

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for epoch in range(n_epochs):
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # Run the training operation and measure the loss:
            _, loss_train = sess.run(
                [training_op, loss],
                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                           y: y_batch,
                           mask_with_labels: True})
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train),
                  end="")

        # At the end of each epoch,
        # measure the validation loss and accuracy:
        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch, y_batch = mnist.validation.next_batch(batch_size)
            loss_val, acc_val = sess.run(
                    [loss, accuracy],
                    feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                               y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_validation,
                      iteration * 100 / n_iterations_validation),
                  end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if loss_val < best_loss_val else ""))

        # And save the model if it improved:
        if loss_val < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val

'''eval'''
n_iterations_test = mnist.test.num_examples // batch_size

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    loss_tests = []
    acc_tests = []
    for iteration in range(1, n_iterations_test + 1):
        X_batch, y_batch = mnist.test.next_batch(batch_size)
        loss_test, acc_test = sess.run(
                [loss, accuracy],
                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                           y: y_batch})
        loss_tests.append(loss_test)
        acc_tests.append(acc_test)
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                  iteration, n_iterations_test,
                  iteration * 100 / n_iterations_test),
              end=" " * 10)
    loss_test = np.mean(loss_tests)
    acc_test = np.mean(acc_tests)
    print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
        acc_test * 100, loss_test))

'''

##############
运行结果：  
##############
[root@master CapsNetSimple]# python  CapsNets_mnist.py 
Extracting /tmp/data/train-images-idx3-ubyte.gz
Extracting /tmp/data/train-labels-idx1-ubyte.gz
Extracting /tmp/data/t10k-images-idx3-ubyte.gz
Extracting /tmp/data/t10k-labels-idx1-ubyte.gz
WARNING:tensorflow:From CapsNets_mnist.py:51: calling reduce_sum (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
WARNING:tensorflow:From CapsNets_mnist.py:90: calling softmax (from tensorflow.python.ops.nn_ops) with dim is deprecated and will be removed in a future version.
Instructions for updating:
dim is deprecated, use axis instead
Epoch: 1  Val accuracy: 99.3000%  Loss: 0.179428 (improved)
Epoch: 2  Val accuracy: 99.3200%  Loss: 0.173912 (improved)
Epoch: 3  Val accuracy: 99.3800%  Loss: 0.171009 (improved)
Epoch: 4  Val accuracy: 99.4000%  Loss: 0.164026 (improved)
Epoch: 5  Val accuracy: 99.4000%  Loss: 0.161676 (improved)
Final test accuracy: 99.4300%  Loss: 0.162981

'''





