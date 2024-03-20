# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w5-1_PerspectiveTransformation.py
import cv2
import numpy as np

'''
cv2.approxPolyDP() 多边形逼近
作用:
对目标图像进行近似多边形拟合，使用一个较少顶点的多边形去拟合一个曲线轮廓，要求拟合曲线与实际轮廓曲线的距离小于某一阀值。

函数原形：
cv2.approxPolyDP(curve, epsilon, closed) -> approxCurve

参数：
curve ： 图像轮廓点集，一般由轮廓检测得到
epsilon ： 原始曲线与近似曲线的最大距离，参数越小，两直线越接近
closed ： 得到的近似曲线是否封闭，一般为True

返回值：
approxCurve ：返回的拟合后的多边形顶点集。
'''


def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

import imutils

def CornerDetection(img):
    # img = cv2.imread('photo1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    edged = cv2.Canny(dilate, 30, 120, 3)  # 边缘检测

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    # cnts = cnts[0]
    # 判断opencv的版本, # 判断是opencv2还是opencv3
    if imutils.is_cv2():
        cnts = cnts[0]
    else:
        cnts = cnts[1]

    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 根据轮廓面积从大到小排序
        for c in cnts:
            peri = cv2.arcLength(c, True)  # 计算轮廓周长
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 轮廓多边形拟合
            # 轮廓为4个点表示找到纸张
            if len(approx) == 4:
                docCnt = approx
                break
    return docCnt


img = cv2.imread('photo1.jpg')
# 角点检测，找出纸张的四个顶点
points = CornerDetection(img)
src = np.squeeze(points).astype(np.float32)
# 根据四个顶点的位置顺序，列出对应目标定点的坐标
dst = np.float32([[0, 0], [0, 640], [480, 640], [480, 0]])
print(img.shape)
# 调用opencv接口生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
# 调用自定义函数实现透视变换
m1 = WarpPerspectiveMatrix(src, dst)
print("warpMatrix m :")
print(m)
print("warpMatrix m1 :")
print(m1)
result = cv2.warpPerspective(img, m, (480, 640))
result1 = cv2.warpPerspective(img, m1, (480, 640))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.imshow("result1", result1)
cv2.waitKey(0)
