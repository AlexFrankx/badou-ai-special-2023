# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w4-1_PCA_detail.py
"""
使用PCA求样本矩阵X的k阶降维矩阵Z
PCA的一般处理步骤分为：
1.中心化
2.求协方差矩阵
3.对协方差矩阵求特征值和特征向量，从而得到降维矩阵
4.通过中心化后的矩阵和降维矩阵的乘积即可得到最终的结果。
"""

# coding=utf-8

import numpy as np


class PCA():
    """
    请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    """
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        print('样本矩阵X:\n', X)

        # 中心化矩阵
        centrX = X - X.mean(axis=0)
        print('样本矩阵X的中心化centrX:\n', centrX)

        # 求协方差矩阵
        C = np.dot(centrX.T, centrX) / (centrX.shape[0] - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)

        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(C)
        print('样本集的协方差矩阵C的特征值:\n', eig_vals)
        print('样本集的协方差矩阵C的特征向量:\n', eig_vectors)

        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        #  求X的降维转换矩阵W, shape = (n, k), n是X的特征维度总数，k是降维矩阵的特征维度
        W = eig_vectors[:, idx[:self.n_components]]
        print('%d阶降维转换矩阵W:\n' % self.n_components, W)

        # 按照Z=XW求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'
        Z = np.dot(X, W)
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z



if __name__ == '__main__':
    '4样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35]])
    k = np.shape(X)[1] - 1
    pca = PCA(n_components=k)
    X_new = pca.fit_transform(X)

