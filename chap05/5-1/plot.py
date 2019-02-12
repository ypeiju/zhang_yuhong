import numpy as np
import matplotlib.pyplot as plt

dataset = [[1.2, 1.1], [2.4, 3.5], [4.1, 3.2], [3.4, 2.8], [5, 5.4]]
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
# red dashes, blue squares and green triangles
plt.axis([0, 6, 0, 6])
plt.plot(x, y, 'bs')
#plt.show
plt.grid()
plt.savefig('scatter.jpg')
