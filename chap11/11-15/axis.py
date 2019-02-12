#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 22:22:45 2018

@author: yhilly
"""

import tensorflow as tf
x1 = tf.constant([     
         [[1, 1, 1], [2, 2, 2]],
         [[3, 3, 3], [4, 4, 4]] 
        ]
)

z0 = tf.reduce_sum(x1, 0)
z1 = tf.reduce_sum(x1, 1) 
z2 = tf.reduce_sum(x1, 2)
z3 = tf.reduce_sum(x1)

with tf.Session() as sess: 
    re0 = sess.run(z0)
    print("==========\n",re0)
    re1 = sess.run(z1)
    print("==========\n",re1)
    re2 = sess.run(z2)
    print("==========\n",re2)
    re3 = sess.run(z3)
    print("==========\n",re3)

    
