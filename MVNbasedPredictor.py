#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import linalg

myfile = open('data.pickle', 'rb')
mydict = pickle.load(myfile)

X_train = mydict['X_train']
X_test = mydict['X_test']
Y_train = mydict['Y_train']
Y_test = mydict['Y_test']

predictive_mean = np.empty(X_test.shape[0])
predictive_std = np.empty(X_test.shape[0])

sigma = 0.1
sigma_f = 1.0
ls = 2.0

N = X_train.size
M = X_test.size
X = np.append(X_train, X_test)
def Sig(i, j):
    return (sigma_f ** 2) * math.exp(-(((X[i] - X[j]) / ls) ** 2) / 2) + (sigma ** 2) * (1 if (i == j) else 0)
I = np.identity(N)
Cov = np.empty((N, N))
Y = np.empty((N, 1))
for i in range(N):
    for j in range(N):
        Cov[i][j] = Sig(i, j)
for i in range(N):
    Y[i] = Y_train[i]
for k in range(M):
    K = np.empty((1, N))
    for i in range(N):
        K[0][i] = Sig(k + N, i)
    predictive_mean[k] = np.linalg.multi_dot([K, linalg.inv(Cov + sigma * sigma * I), Y])
    predictive_std[k] = (Sig(k, k) - np.linalg.multi_dot([K, linalg.inv(Cov + sigma * sigma * I), K.T])) ** (1/2)

fig = plt.figure()
plt.plot(X_train, Y_train, linestyle='', color='b', markersize=5, marker='+',label="Training data")
plt.plot(X_test, Y_test, linestyle='', color='orange', markersize=2, marker='^',label="Testing data")
plt.plot(X_test, predictive_mean, linestyle=':', color='green')
plt.fill_between(X_test.flatten(), predictive_mean - predictive_std, predictive_mean + predictive_std, color='green', alpha=0.13)
plt.fill_between(X_test.flatten(), predictive_mean - 2*predictive_std, predictive_mean + 2*predictive_std, color='green', alpha=0.07)
plt.fill_between(X_test.flatten(), predictive_mean - 3*predictive_std, predictive_mean + 3*predictive_std, color='green', alpha=0.04)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
