#!/usr/bin/env python3
# Author: Armit
# Create Time: 2020年4月6日

import numpy as np
import matplotlib.pyplot as plt

x = np.array([i + np.random.normal() for i in np.linspace(0, 10, 101)])
y = np.array([(2 * i - 1) + np.random.normal(0, 3) for i in x])
#plt.scatter(x, y, c='g')

X = np.vstack((x, y))
N = X.shape[1]
print('Shape of X: ', X.shape)

demeanX = X - X.mean(axis=1).repeat(N).reshape(X.shape)
x1, y1 = demeanX
plt.scatter(x1, y1, c='b')

covX = np.dot(demeanX, demeanX.T) / N     # 也可以不除以N
val, vec = np.linalg.eig(covX)
print(val)
print(vec)

x0 = np.linspace(-5, 5, 101)
for v in vec:
  k = - v[1] / v[0]
  print('k = ', k)
  y0 = k * x0
  plt.plot(x0, y0, c='r')

plt.xlim((-10, 10))
plt.ylim((-10, 10))
plt.show()

