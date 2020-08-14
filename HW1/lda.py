#!/usr/bin/env python3
# Author: Armit
# Create Time: 2020年4月6日

import numpy as np
import matplotlib.pyplot as plt

N = 50
x, y = [[],[]], [[],[]]
for i, c in enumerate([(2,5), (4,3)]):
  for _ in range(N):
    x[i].append(c[0] + np.random.normal())
    y[i].append(c[1] + np.random.normal())

x, y = [np.array(i) for i in x], [np.array(i) for i in y]
plt.scatter(x[0], y[0], c='b')
plt.scatter(x[1], y[1], c='r')

X0 = np.array([np.array(p) for p in zip(x[0], y[0])]).T
meanX0 = X0.mean(axis=1).reshape((2,1))
X1 = np.array([np.array(p) for p in zip(x[1], y[1])]).T
meanX1 = X1.mean(axis=1).reshape((2,1))
demeanX0 = X0 - meanX0.repeat(N).reshape(2, N)
demeanX1 = X1 - meanX1.repeat(N).reshape(2, N)
A = np.dot(demeanX0, demeanX0.T) + np.dot(demeanX1, demeanX1.T)
B = meanX0 - meanX1
w = np.dot(np.linalg.inv(A), B)

x0 = np.linspace(-2, 8, 101)
k = w[1] / w[0]
print('k = ', k)
y0 = k * x0
plt.plot(x0, y0, c='r')

plt.show()

