# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w4-2_PCA.py


import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

import numpy as np

if __name__ == '__main__':
    '4样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35]])
    K = np.shape(X)[1] - 1
    print('样本集(4行3列，10个样例，每个样例3个特征):\n', X)
    pca = dp.PCA(n_components=2)  # 加载pca算法，设置降维后主成分数目为2
    reduced_x = pca.fit_transform(X)  # 对原始数据进行降维，内部使用svd进行降维，保存在reduced_x中
    print('样本矩阵X的降维矩阵Z:\n', reduced_x)

# 鸢尾花数据做PCA
x, y = load_iris(return_X_y=True)  # 加载数据，x表示数据集中的属性数据，y表示数据标签
pca = dp.PCA(n_components=2)  # 加载pca算法，设置降维后主成分数目为2
reduced_x = pca.fit_transform(x)  # 对原始数据进行降维，保存在reduced_x中
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_x)):  # 按鸢尾花的类别将降维后的数据点保存在不同的表中
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x, red_y, c='r', marker='x') # 生成一个scatter散点图
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
