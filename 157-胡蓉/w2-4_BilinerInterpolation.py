# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w2-4_NearestNeighborInterpolation.py


import numpy as np
import cv2
import warnings

warnings.filterwarnings('ignore')  # 忽略一些warning内容，无需打印


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_w, scale_h = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 将目标坐标系映射到原始坐标系
                # 通过平移的方式将原始图像与目标图像中心点对齐
                # 直接映射公式为src_x = dst_x * scale_x，通过推到原始与目标坐标同时平移0.5可对齐中心点
                src_x = (dst_x + 0.5) * scale_w - 0.5
                src_y = (dst_y + 0.5) * scale_h - 0.5

                # 分别计算x y方向插值的两个点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 双线性插值公式计算出目标点
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (640, 480))
    cv2.imshow("img_org", img)
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()
