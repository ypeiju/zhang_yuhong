#样本数据
dataset = [[1.2, 1.1], [2.4, 3.5], [4.1, 3.2], [3.4, 2.8], [5, 5.4]]
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]

# 计算样本均值的函数
def mean(values):
	return sum(values) / float(len(values))

# 计算 x 与 y协方差的函数
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

#获取均值
mean_x, mean_y = mean(x), mean(y)
#获取协方差
covar = covariance(x, mean_x, y, mean_y)
#输出协方差
print('协方差 = : %.3f' % (covar))
