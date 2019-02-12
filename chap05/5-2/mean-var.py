#定义求均值的函数
def mean(values):
	return sum(values) / float(len(values))
 
# 计算求方差的函数
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])
 
# 开始计算均值和方差
dataset = [[1.2, 1.1], [2.4, 3.5], [4.1, 3.2], [3.4, 2.8], [5, 5.4]]
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
mean_x, mean_y = mean(x), mean(y)
var_x, var_y = variance(x, mean_x), variance(y, mean_y)
print('x 统计特性: 均值 = %.3f 方差 = %.3f' % (mean_x, var_x))
print('y 统计特性: 均值 = %.3f 方差 = %.3f' % (mean_y, var_y))
