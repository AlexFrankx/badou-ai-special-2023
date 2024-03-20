# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w2-2_NearestNeighborInterpolation.py


import numpy as np
import cv2
import warnings

warnings.filterwarnings('ignore')  # 忽略一些warning内容，无需打印
# opencv读取图片
img = cv2.imread("lenna.png")  # opencv读取图片，存储顺序为BGR
h, w, c = img.shape  # 获取图片的high和wide, img.shape结果为（h,w,c）

w1 = 640
h1 = 480
img_nearest = np.zeros([h1, w1, c], img.dtype)  # 创建一张640*480单通道图片

rw = float(w) / w1
rh = float(h) / h1
for i in range(h1):
    for j in range(w1):
        # 将目标坐标缩放到原图坐标系上，然后采用最近邻算法，取四舍五入对应坐标的值，防止坐标值超界，取计算坐标值与原图高（或宽）的最小值
        src_x = min(int(i * rh + 0.5), h - 1)
        src_y = min(int(j * rw + 0.5), w - 1)
        img_nearest[i, j] = img[src_x, src_y]  # 取出对应的像素值
cv2.imshow("img_org", img)
cv2.imshow("img_nearest", img_nearest)
cv2.waitKey(0)
