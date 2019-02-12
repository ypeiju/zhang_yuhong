#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 05:32:35 2017

@author: yhilly
"""

import tensorflow as tf
   
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    d = tf.constant([[1.,2.,3.,4.],
                     [5.,6.,7.,8.],
                     [9.,10.,11.,12.],
                     [13.,14.,15.,16.]])
    print(sess.run(tf.shape(d)))
    print(sess.run(d))
    print("-----------------------\n")
    #由于[4,4] == [4,4] 行和列都为独立
    dropout_a44 = tf.nn.dropout(d, 0.5)
    result_dropout_a44 = sess.run(dropout_a44)
    print(result_dropout_a44)
    print("-----------------------\n")
    #noise_shpae[0]=4 == tf.shape(d)[0]=4  
    #noise_shpae[1]=4 != tf.shape(d)[1]=1
    #所以[0]即行独立，[1]即列相关，每个行同为0或同不为0
    dropout_a41 = tf.nn.dropout(d, 0.5, noise_shape = [4,1])
    result_dropout_a41 = sess.run(dropout_a41)
    print(result_dropout_a41)
    print("-----------------------\n")
    #noise_shpae[0]=1 ！= tf.shape(d)[0]=4  
    #noise_shpae[1]=4 == tf.shape(d)[1]=4
    #所以[1]即列独立，[0]即行相关，每个列同为0或同不为0
    dropout_a24 = tf.nn.dropout(d, 0.5, noise_shape = [1,4])
    result_dropout_a24 = sess.run(dropout_a24)
    print(result_dropout_a24)
    #不相等的noise_shape只能为1