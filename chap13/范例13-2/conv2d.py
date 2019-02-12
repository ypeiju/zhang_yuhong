import tensorflow as tf

# 输入数据（可以是一副图像）
temp = tf.constant([0, 1, 0, 1, 2, 1, 1, 0, 3, 1, 1, 0, 4, 4, 5, 4], tf.float32)
temp2 = tf.reshape(temp, [2, 2, 2, 2])

# 卷积核
filter = tf.constant([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tf.float32)
filter2 = tf.reshape(filter, [2, 2, 2, 2])

# 在4D矩阵上执行卷积操作
convolution = tf.nn.conv2d(temp2, filter2, [1, 1, 1, 1], padding="SAME")

# 初始化回话
session = tf.Session()
tf.global_variables_initializer()

# 计算所有值.
print("输入数据：")
print(session.run(temp2))
print("卷积核：")
print(session.run(filter2))
print("卷积特征图：")
print(session.run(convolution))
