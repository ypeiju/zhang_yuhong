#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:37:25 2018

@author: yhilly
"""

from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
   
class Database():
    def __init__(self, db_file):   
       self.filename = db_file
       self.dataset = list()

    # 导入CSV 文件
    def load_csv(self):
        with open(filename, 'r') as file:
            csv_reader = reader(file)   
            for row in csv_reader:
                if not row:   #判定是否有空行，如有，则跳入到下一行
                    continue
                self.dataset.append(row)
#        print(self.dataset)
              
    #将n-1列的属性字符串列转换为浮点数，第n列为分类的类别           
    def dataset_str_to_float(self):
       col_len = len(self.dataset[0]) - 1
       for row in self.dataset:
           for column in range(col_len):            
               row[column] = float(row[column].strip())
                
    # 将最后一列（n）的类别，转换为整型，并提取有多少个类
    def str_column_to_int(self, column):    
       class_values = [row[column] for row in self.dataset]   #读取指定列的数字
       unique = set(class_values)    #用集合来合并类
       lookup = dict()
       for i, value in enumerate(unique):
           lookup[value] = i
       for row in self.dataset:
           row[column] = lookup[row[column]]
    # 找到每一列（属性）的最小和最大值
    def dataset_minmax(self):
       self.minmax = list()
       self.minmax = [[min(column), max(column)] for column in zip(*self.dataset)]

    # 将数据集合中的每个（列）属性都规整化到0-1
    def normalize_dataset(self):
       self.dataset_minmax()
       for row in self.dataset:
           for i in range(len(row)-1):
               row[i] = (row[i] - self.minmax[i][0]) / (self.minmax[i][1] - self.minmax[i][0])
                
    def get_dataset(self):
       # 构建训练数据
       self.load_csv()
       self.dataset_str_to_float()
       self.str_column_to_int(len(self.dataset[0])-1)
       self.normalize_dataset()
       return self.dataset
           
class BP_Network():
    # 初始化神经网络
    def __init__(self, n_inputs,n_hidden,n_outputs):        
       self.n_inputs = n_inputs
       self.n_hidden = n_hidden    
       self.n_outputs = n_outputs
       self.network = list()
       hidden_layer = [{'weights':[random() for i in range(self.n_inputs + 1)]} for i in range(self.n_hidden)]
       self.network.append(hidden_layer)
       output_layer = [{'weights':[random() for i in range(self.n_hidden + 1)]} for i in range(self.n_outputs)]
       self.network.append(output_layer)

    # 计算神经元的激活值（加权之和）
    def activate(self, weights, inputs):
       activation = weights[-1]
       for i in range(len(weights)-1):
           activation += weights[i] * inputs[i]
       return activation

    # 定义激活函数
    def transfer(self, activation):
       return 1.0 / (1.0 + exp(-activation))

    # 计算神经网络的正向传播
    def forward_propagate(self,  row):
       inputs = row
       for layer in self.network:
           new_inputs = []
           for neuron in layer:
               activation = self.activate(neuron['weights'], inputs)
               neuron['output'] = self.transfer(activation)
               new_inputs.append(neuron['output'])
           inputs = new_inputs
       return inputs

    # 计算激活函数的导数
    def transfer_derivative(self,output):
       return output * (1.0 - output)

    # 反向传播误差信息，并将纠偏责任存储在神经元中
    def backward_propagate_error(self, expected):
       for i in reversed(range(len(self.network))):
           layer = self.network[i]
           errors = list()
           if i != len(self.network)-1:
               for j in range(len(layer)):
                   error = 0.0
                   for neuron in self.network[i + 1]:
                       error += (neuron['weights'][j] * neuron['responsibility'])
                   errors.append(error)
           else:
               for j in range(len(layer)):
                   neuron = layer[j]
                   errors.append(expected[j] - neuron['output'])
           for j in range(len(layer)):
               neuron = layer[j]
               neuron['responsibility'] = errors[j] * self.transfer_derivative (neuron['output'])

    # 根据误差，更新网络权重
    def _update_weights(self, row):
       for i in range(len(self.network)):
           inputs = row[:-1]
           if i != 0:
               inputs = [neuron['output'] for neuron in self.network[i - 1]]
           for neuron in self.network[i]:
               for j in range(len(inputs)):
                   neuron['weights'][j] += self.l_rate * neuron['responsibility'] * inputs[j]
               neuron['weights'][-1] +=  self.l_rate * neuron['responsibility']

    # 根据指定的训练周期训练网络
    def train_network(self, train):
       for epoch in range(self.n_epoch):
           sum_error = 0
           for row in train:
               outputs = self.forward_propagate(row)
               expected = [0 for i in range(self.n_outputs)]
               expected[row[-1]] = 1
               sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
               self.backward_propagate_error(expected)
               self._update_weights(row)
           print('>周期=%d, 误差=%.3f' % (epoch, sum_error))

    #利用训练好的网络，预测“新”数据
    def predict(self, row):
       outputs = self.forward_propagate(row)
       return outputs.index(max(outputs))

    # 利用随机梯度递减策略，训练网络
    def back_propagation(self,train, test):
       self.train_network(train)
       predictions = list()
       for row in test:
           prediction = self.predict(row)
           predictions.append(prediction)
       return(predictions)
    # 将数据库分割为 k等份
    def cross_validation_split(self, n_folds):
       dataset_split = list()
       dataset_copy = list(self.dataset)
       fold_size = int(len(self.dataset) / n_folds)

       for i in range(n_folds):
           fold = list()
           while len(fold) < fold_size:
               index = randrange(len(dataset_copy))
               fold.append(dataset_copy.pop(index))
           dataset_split.append(fold)
       return dataset_split

    # 用预测正确百分比来衡量正确率
    def accuracy_metric(self, actual, predicted):
       correct = 0
       for i in range(len(actual)):
           if actual[i] == predicted[i]:
               correct += 1
       return correct / float(len(actual)) * 100.0

    #用每一个交叉分割的块（训练集合，试集合）来评估BP算法
    def evaluate_algorithm(self, dataset, n_folds, l_rate, n_epoch):
       self.l_rate = l_rate
       self.n_epoch = n_epoch
       self.dataset = dataset
       folds = self.cross_validation_split(n_folds)
       scores = list()
       for fold in folds:
           train_set = list(folds)
           train_set.remove(fold)
           train_set = sum(train_set, [])
           test_set = list()
           for row in fold:
               row_copy = list(row)
               test_set.append(row_copy)
               row_copy[-1] = None
           predicted = self.back_propagation(train_set, test_set)
           actual = [row[-1] for row in fold]
           accuracy = self.accuracy_metric(actual, predicted)
           scores.append(accuracy)
       return scores

if __name__ == '__main__':     
    #设置随机种子
    seed(2)  
    # 构建训练数据
    filename = 'seeds_dataset.csv'   
    db = Database(filename) 
    dataset = db.get_dataset()
    # 设置网络初始化参数
    n_inputs = len(dataset[0]) - 1
    n_hidden = 5
    n_outputs = len(set([row[-1] for row in dataset]))
    BP = BP_Network(n_inputs,n_hidden,n_outputs)
    l_rate = 0.3
    n_folds = 5
    n_epoch = 500
    scores = BP.evaluate_algorithm(dataset, n_folds, l_rate, n_epoch)
    print('评估算法正交验证得分: %s' % scores)
    print('平均准确率: %.3f%%' % (sum(scores)/float(len(scores))))
