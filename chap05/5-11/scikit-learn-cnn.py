import numpy as np
import pandas as pd
from sklearn import datasets

#加载IRIS数据集合
scikit_iris = datasets.load_iris()
#转换为pandas的DataFrame格式，以便于观察数据
pd_iris = pd.DataFrame(
data = np.c_[scikit_iris['data'], scikit_iris['target']],
columns = np.append(scikit_iris.feature_names, ['y']))
#print(pd_iris.head(2))

#选择全部特征参与训练模型
X = pd_iris[scikit_iris.feature_names]
y = pd_iris['y']

#（1）选择模型
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
#（2）拟合模型（训练模型）
knn.fit(X,y)
#（3）预测新数据
knn.predict([[4,3,5,3]])
#print(knn.predict([[4,3,5,3]]))
