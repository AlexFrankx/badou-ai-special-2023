# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w3-1_HistogramEqualization.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import warnings

warnings.filterwarnings('ignore')  # 忽略一些warning内容，无需打印

# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度直方图统计
hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure()
plt.plot(hist,color='b')
plt.show()

#原始灰度直方图绘制
plt.figure()
plt.subplot(2, 1, 1)
plt.title("原始灰度直方图", fontsize=10)
plt.hist(gray.ravel(), 256)

# 灰度图像直方图均衡化绘制
dst = cv2.equalizeHist(gray)
plt.subplot(2, 1, 2)
plt.title("灰度图像直方图均衡化", fontsize=10)
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("orggray and Histogram Equalization", np.hstack([gray, dst]))




# 彩色图像直方图均衡化
# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("orgrgb and Histogram Equalization",  np.hstack([img, result]))
cv2.waitKey(0)
