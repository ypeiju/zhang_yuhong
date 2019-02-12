from random import seed
from random import randrange
from csv import reader
from math import sqrt
 
# 导入CSV 文件
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        #读取表头X，Y
        headings = next(csv_reader)
        #文件指针下一至第一条真正数据        
        for row in csv_reader:
            if not row:   #判定是否有空行，如有，则跳入到下一行
                continue
            dataset.append(row)
    return dataset

#将字符串列转换为浮点数
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

#将数据集分割为训练集合测试集合两部分
def train_test_split(dataset, percent):
    train = list()
    train_size = percent * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy

# 计算样本均值的函数
def mean(values):
    return sum(values) / float(len(values))

# 计算 x 与 y协方差的函数
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

#计算方差的函数
# Calculate the variance of a list of numbers
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

# 计算回归系数的函数
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    w1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    w0 = y_mean - w1 * x_mean
    return (w0, w1)

#计算均方根误差RMSE
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

# 构建简单线性
def simple_linear_regression(train, test):
    predictions = list()
    w0, w1 = coefficients(train)
    for row in test:
        y_model =  w1 * row[0] + w0
        predictions.append(y_model)
    return predictions
    
#使用分割开的训练集合测试集运行评估算法
def evaluate_algorithm(dataset, algorithm, split_percent, *args):
    train, test = train_test_split(dataset, split_percent)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]
    rmse = rmse_metric(actual, predicted)
    return rmse

#设置随机数种子，为随机挑选训练和测试数据集做准备
seed(2)
# 导入保险数据并做数据分割准备
filename = 'insurance.csv'
dataset = load_csv(filename)
for col in range(len(dataset[0])):
    str_column_to_float(dataset, col)

# 设置数据集合分割百分比
percent = 0.6
rmse = evaluate_algorithm(dataset, simple_linear_regression, percent)
print('RMSE: %.3f' % (rmse))
