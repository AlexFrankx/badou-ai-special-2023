# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w4-4_canny.py
import cv2


'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是滞后阈值1；
第三个参数是滞后阈值2。
'''

gray = cv2.imread("lenna.png", 0)
cv2.imshow("canny", cv2.Canny(gray, 50, 150))
cv2.waitKey()
cv2.destroyAllWindows()
