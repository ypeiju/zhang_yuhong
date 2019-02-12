# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:08:42 2017

@author: Yuhong
"""

from random import seed
from random import random
 
# 初始化网络
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

if __name__ == '__main__':   
    seed(1)
    network = initialize_network(2, 2, 2)
    for layer in network:
        print(layer)