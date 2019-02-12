#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 05:38:27 2017

@author: yhilly
"""

import tensorflow as tf
 
layer_input = tf.constant([
        [
            [[1.0],[2.0],[1.0],[6.0]],
            [[7.0],[4.0],[5.0],[8.0]],
            [[4.0],[2.0],[2.0],[0.0]],
            [[1.0],[5.0],[1.0],[4.0]]
        ]
    ])

batch_size      = 1
input_height    = 2
input_width     = 2
input_channels  = 1
ksize = [batch_size, input_height, input_width, input_channels]
pooling=tf.nn.max_pool(layer_input,ksize,[1,2,2,1],padding='VALID')

with tf.Session() as sess:
    print("origin_data:")
    image = sess.run(layer_input)
    print (image)
    print("pool_reslut:")
    result = sess.run(pooling)
    print (result)