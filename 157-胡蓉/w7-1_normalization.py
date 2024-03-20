# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w7-1_normalization.py

import numpy as np
import matplotlib.pyplot as plt


# 归一化的三种方式

def minmaxNormalization(x):
    """
    归一化（0~1）
    x_=(x−x_min)/(x_max−x_min)
    """
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


def meanNormalization(x):
    """
    归一化（-1~1）
    x_=(x−x_mean)/(x_max−x_min)
    """
    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]


# 标准化
def z_scoreNormalization(x):
    """
    x_=(x−μ)/σ
    """
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    s = s2 ** 0.5
    return [(i - x_mean) / s for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

n1 = minmaxNormalization(l)
n2 = meanNormalization(l)
z = z_scoreNormalization(l)
print(n1)
print(n2)
print(z)

'''
蓝线为原始数据，橙线为z
'''
# 计算原始数据分布情况,进行z-score标准化后
cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)
plt.plot(l, cs)
plt.plot(z, cs)
plt.show()

x = []
for i in range(len(l)):
    x.append(i)
plt.scatter(x, l, c='r')
plt.scatter(x, n1, c='g')
plt.scatter(x, n2, c='b')
plt.legend(["l", "n1", "n2"])
plt.show()
