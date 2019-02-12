#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 03:41:49 2017

@author: yhilly
"""

import tensorflow as tf
features = tf.range(-1,3)
features2 = tf.to_float(features)

with tf.Session() as sess:
   print(sess.run([features2,tf.tanh(features2)]))