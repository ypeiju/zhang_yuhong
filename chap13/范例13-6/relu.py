#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 03:46:14 2017

@author: yhilly
"""


import tensorflow as tf
features = tf.range(-3,4)

with tf.Session() as sess:
    feature = sess.run(features)
    result_relu_feature = sess.run(tf.nn.relu(features))
    print('feature :{} \nrelu(feature): {}'.format(feature,result_relu_feature))