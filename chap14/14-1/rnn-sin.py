import random
import numpy as np
import tensorflow as tf
truncated_backprop_length = 10
state_size = 10
batch_size = 10

def generateData(n): 
    x_seq=[]
    y_seq=[]
    for i in range(2000):
        k = random.uniform(1,50) 
        x = [np.sin(k + j) for j in range(0, n)]
        y = [np.sin(k + n)]
        x_seq.append(x)
        y_seq.append(y)

    train_x = np.array (x_seq [0 : 1500] )
    train_y = np.array (y_seq [0 : 1500] )
    test_x  = np.array (x_seq [1500: ] )
    test_y  = np.array (y_seq [1500: ] )
    return train_x,train_y,test_x,test_y

(train_x, train_y, test_x, test_y) = generateData(truncated_backprop_length )


X_placeholder = tf.placeholder(tf.float32, [None, truncated_backprop_length])

Y_placeholder = tf.placeholder(tf.float32, [None, 1])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

W = tf.Variable(np.random.rand(state_size + 1, state_size), dtype=tf.float32)

b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, 1),dtype=tf.float32)

b2 = tf.Variable(np.zeros(1), dtype = tf.float32)

inputs_series = tf.unstack(X_placeholder, axis = 1)

current_state = init_state
for current_input in inputs_series:
    current_input = tf.reshape(current_input, [batch_size, 1]) 
    input_and_state_concatenated = tf.concat( [current_input, current_state], 1) 
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b) 
    current_state = next_state
    
logits  = tf.matmul(current_state, W2) + b2 
loss    = tf.square(tf.subtract(Y_placeholder,logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _current_state = np.zeros((batch_size, state_size))
    
    for epoch_id in range(500):
        for batch_id in range(len(train_x) // batch_size):
            begin = batch_id * batch_size 
            end = begin + batch_size 
            
            batchX = train_x[begin:end]
            batchY = train_y[begin:end]
            
            _train_step, _current_state = sess.run(
                [train_step, current_state],
                feed_dict={
                    X_placeholder:batchX,
                    Y_placeholder:batchY,
                    init_state:_current_state
                })
    
        test_indices = np.arange(len(test_x)) 
        np.random.shuffle(test_indices)
        test_indices = test_indices[0 : 10] 
        
        x = test_x[test_indices]
        y = test_y[test_indices]
        val_loss = np.mean(sess.run(
            loss,
            feed_dict = {
                X_placeholder : x,
                Y_placeholder : y,
                init_state : _current_state
            })) 
        print('epoch: %s'%epoch_id,', loss: %g'%val_loss)

    writer = tf.summary.FileWriter('./my_graph/1')
    writer.add_graph(sess.graph)