from statistics import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

data = pd.read_csv('data_linear.csv')

x = np.array([data.values[: , 0]])
y = np.array([data.values[: , 1]])
x = x.T
y = y.T
plt.plot(x, y, 'b.')
one = np.ones((x.shape[0], 1))
xbar = np.concatenate((one, x), axis = 1)
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(xbar, y)
w = regr.coef_
A = w[0][0]
B = w[0][1]
ax = np.linspace(min(x), max(x), 2)
ay = A + B*ax
plt.plot(ax, ay, color = 'red')
print(A, B)
plt.show()