# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w6-3_Hash.py
import cv2
import numpy as np
from skimage import util


# 均值哈希算法
def meanHash(img):
    # 缩放为8*8
    img = cv2.resize(img, (8, 8))
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 求平均灰度
    mean = gray.mean()

    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > mean:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 差值算法
def diffHash(img):
    # 缩放8*9
    img = cv2.resize(img, (9, 8))
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def min_max_normalize(x):
    """图像归一化，并转换倒0-255之间"""
    x_min = np.min(x)
    x_max = np.max(x)
    x_nor = (x - x_min) / (x_max - x_min)
    r_img = (x_nor * 255).astype(np.uint8)
    return r_img


# 原图
img1 = cv2.imread('lenna.png')

# 加入高斯噪声图
gaussian_img = util.random_noise(img1, mode='gaussian')
img2 = min_max_normalize(gaussian_img)

hash1 = meanHash(img1)
hash2 = meanHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n)

hash1 = diffHash(img1)
hash2 = diffHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('差值哈希算法相似度：', n)
