#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 03:49:07 2018

@author: yhilly
"""

import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[1,1])
m = tf.matmul(x,x)

with tf.Session() as sess:
    m_out = sess.run(m, feed_dict = {x:[[2.0]]})

print(m_out)