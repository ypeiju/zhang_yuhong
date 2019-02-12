#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 01:35:48 2017

@author: yhilly
"""

import tensorflow as tf
with tf.Session() as sess:
    print("tanh(0.2):", sess.run(tf.tanh(0.2)))
    print("tanh(2.0):", sess.run(tf.tanh(2.0)))
    print("tanh(20.0):", sess.run(tf.tanh(20.0)))