# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 16:39:09 2017

@author: Yuhong
"""
from csv import reader
class LinearUnit(object):
    def __init__(self, input_para_num, acti_func):
        '''
        初始化线性单元激活函数。
        '''
        self.activator = acti_func    
                       
    def __str__(self):
        '''
        打印学习到的权重，其中w0为偏置项
        '''
        str1 = "final weights\n\t"
        for i in range(len(self.weights)):
            str1 =+ str(self.weights[i]) +"\n\t" 
        return str1
        
    def predict(self, row_vec):
        '''
        输入向量，输出线性单元的计算结果
        '''
        act_values = self.weights[ 0 ]
        for i in range(len(row_vec) - 1):
            act_values += self.weights[ i + 1 ] * row_vec [ i ]
        return self.activator(act_values)
    
    def train_sgd(self, dataset, rate, n_epoch):
        '''
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        '''
        # 权重向量初始化为0  
        self.weights = [0.0 for i in range(len(dataset[0]))]
        for i in range(n_epoch):
            for input_vec_label in dataset:
                prediction = self.predict(input_vec_label)
                # 更新权重
                self._update_weights(input_vec_label,prediction, rate)
                
                  
    def _update_weights(self,input_vec_label,prediction, rate):
        delta =   prediction -input_vec_label[-1]               
        #更新权值，第一个元素：哑元的权值
        self.weights[ 0 ]  =   self.weights[ 0 ] - rate * delta  
        for i in range(len(self.weights) - 1):
           self.weights[ i + 1 ] = self.weights[ i + 1 ] - rate * delta * input_vec_label[ i ]
# 定义激活函数f      
def func_activator(input_value):
    return input_value
    
class Database():
    def __init__(self):       
        self.dataset = list()

    # 导入CSV 文件
    def load_csv(self, filename):
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            #读取表头X，Y
            headings = next(csv_reader)
            #文件指针下一至第一条真正数据        
            for row in csv_reader:
                if not row:   #判定是否有空行，如有，则跳入到下一行
                    continue
                self.dataset.append(row)
        
    #将字符串列转换为浮点数            
    def dataset_str_to_float(self):
        col_len = len(self.dataset[0])
        for row in self.dataset:
            for column in range(col_len):            
                row[column] = float(row[column].strip())
               
    # 找到每一列（属性）的最小和最大值
    def _dataset_minmax(self):
        self.minmax = list()
        for i in range(len(self.dataset[0])):
            col_values = [row[i] for row in self.dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            self.minmax.append([value_min, value_max])
    # 将数据集合中的每个（列）属性都规整化到0-1
    def normalize_dataset(self):
        self._dataset_minmax()
        for row in self.dataset:
            for i in range(len(row)):
                row[i] = (row[i] - self.minmax[i][0]) / (self.minmax[i][1] - self.minmax[i][0])
        return self.dataset
            
def get_training_dataset():
    # 构建训练数据
    db = Database() 
    db.load_csv("winequality-white.csv")
    db.dataset_str_to_float()
    dataset = db.normalize_dataset() 
    return dataset

def train_linear_unit():
 
    '''
    使用数据训练线性单元
    '''
    dataset = get_training_dataset()
    l_rate = 0.01
    n_epoch = 100    
    # 创建训练线性单元，输入参数的特征数为12
    linear_unit = LinearUnit(len(dataset[0]), func_activator)  
    # 训练，迭代100轮, 学习速率为0.01
    linear_unit.train_sgd(dataset, l_rate, n_epoch)  
    #返回训练好的线性单元
    return linear_unit    

if __name__ == '__main__': 
    #获取数据并训练
    LU = train_linear_unit()   
#    # 打印训练获得的权重
    print ("weights = ", LU.weights)
    # 测试
    test_data = [[0.23,0.08,0.20,0.01,0.14,0.07,0.17,0.07,0.55,0.28,0.47,0.67], 
            [0.25,0.10,0.21,0.01,0.11,0.13,0.23,0.08,0.54,0.15,0.47,0.50], 
            [0.28,0.15,0.19,0.02,0.11,0.10,0.20,0.11,0.55,0.49,0.44,0.83], 
            [0.35,0.16,0.17,0.15,0.12,0.07,0.22,0.18,0.37,0.15,0.24,0.33]]
    for i in range(len(test_data)):
        pred = LU.predict(test_data[i])
        print("expected ={0}, predicted = {1}".format(test_data[i][-1], pred))
        