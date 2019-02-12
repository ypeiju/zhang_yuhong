#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:41:41 2018

@author: yhilly
"""
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

a = tf.constant(12)
counter = 0
while not tf.equal(a, 1):
    if tf.equal(a % 2, 0):
        a = a / 2
    else:
        a = 3 * a + 1
    print(a)