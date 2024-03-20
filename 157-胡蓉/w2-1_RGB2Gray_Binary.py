# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w2-1_RGB2Gray_Binary.py

"""
彩色图像的灰度化、二值化
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings

warnings.filterwarnings('ignore')  # 忽略一些warning内容，无需打印


# plt读取图片，plt显示
img = plt.imread("lenna.png")
plt.subplot(221)
plt.imshow(img)
print("---image lenna----")
print(img)

# skimage库实现灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

# numpy实现二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
print("-----image binary------")
print(img_binary)
print(img_binary.shape)
plt.show()


# opencv读取图片，用公式实现灰度化，opencv显示
img = cv2.imread("lenna.png")  # opencv读取图片，存储顺序为BGR
h, w = img.shape[:2]  # 获取图片的high和wide, img.shape结果为（h,w,c）
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        # gray公式为Gray=0.3R+0.59G+0.11B
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像
print("---image gray----")
print(img_gray)
print("image show gray: %s" % img_gray)
cv2.imshow("image show gray", img_gray)

k = cv2.waitKey(0) & 0xFF
if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()


# opencv接口实现灰度化
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# opencv接口进行二值化处理
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Original Image", img)
cv2.imshow("Gray Image", gray_image)
cv2.imshow("Binary Image", binary_image)

k = cv2.waitKey(0) & 0xFF
if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()


# for循环实现二值化，plt显示
binary = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] <= 127:
            binary[i, j] = 0
        else:
            binary[i, j] = 1
print("-----binary------")
print(binary)
plt.imshow(binary, cmap='gray')
plt.show()
