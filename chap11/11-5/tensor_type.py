#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 19:01:04 2018

@author: yhilly
"""

import tensorflow as tf
import numpy as np

n0 = np.array(20, dtype = np.int32)
n1 = np.array([b"Tensor", b"flow", b"is", b"great"])
n2 = np.array([[True, False, False],
                [False, True,False]], 
                dtype = np.bool)
tensor0D = tf.Variable(n0, name = "t_0")
tensor1D = tf.Variable(n1, name = "t_1")
tensor2D = tf.Variable(n2, name = "t_2")
init_Op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_Op)
    print(sess.run(tensor0D))
    print(sess.run(tensor1D))
    print(sess.run(tensor2D))