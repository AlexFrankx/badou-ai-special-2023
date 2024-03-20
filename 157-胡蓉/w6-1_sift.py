# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w6-1_sift.py

"""sift算法已经申请专利，开源OpenCV没有版权，新的OpenCV去掉了这个算法
pip uninstall opencv-python
#推荐使用豆瓣python源
pip install opencv-python==3.4.2.16 -i "https://pypi.doubanio.com/simple/"
pip install opencv-contrib-python==3.4.2.16 -i "https://pypi.doubanio.com/simple/"

"""
import cv2
import numpy as np


def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])  # 解析出 相似的一对 点的坐标
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255),3)  # 将两个相似点连接起来

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


# img1_gray = cv2.imread("iphone1.png")
# img2_gray = cv2.imread("iphone2.png")
img1_gray = cv2.imread("t1.jpg")
img2_gray = cv2.imread("t2.jpg")

sift = cv2.xfeatures2d.SIFT_create()  # sift为实例化的sift函数

# 检测关键点并计算
# kp:关键点信息，包括位置，尺度，方向信息
# des:关键点描述符，每个关键点对应128个梯度信息的特征向量
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# BFmatcher with default parms
bf = cv2.BFMatcher(cv2.NORM_L2)  # 建立匹配关系
matches = bf.knnMatch(des1, des2, k=2)  # 匹配描述子,返回k个最佳匹配

goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()
