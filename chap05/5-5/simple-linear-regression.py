import matplotlib.pyplot as plt
from math import sqrt
#样本数据
dataset = [[1.2, 1.1], [2.4, 3.5], [4.1, 3.2], [3.4, 2.8], [5, 5.4]]
predict_data = list()

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
    
def evaluate_algorithm(dataset, algorithm):
    test_set = list()
    for row in dataset:
        row_copy = list(row)
        row_copy[1] = None
        test_set.append(row_copy)

    global predict_data
    predict_data = algorithm(dataset, test_set)
    for val in predict_data:
        print('%.3f\t' %(val))

    actual = [row[-1] for row in dataset]
    rmse = rmse_metric(actual, predict_data)
    return rmse

# 返回RMSE
rmse = evaluate_algorithm(dataset, simple_linear_regression)
print('RMSE: %.3f' % (rmse))

x = [row[0] for row in dataset]
y = [row[1] for row in dataset]

