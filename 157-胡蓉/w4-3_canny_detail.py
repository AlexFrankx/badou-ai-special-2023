# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w4-3_canny_detail.py
"""
Canny边缘检测算法步骤
1. 对图像进行灰度化
2. 对图像进行高斯滤波：
根据待滤波的像素点及其邻域点的灰度值按照一定的参数规则进行加权平均。这样
可以有效滤去理想图像中叠加的高频噪声。
3. 检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）。
4. 对梯度幅值进行非极大值抑制：通俗意义上是指寻找像素点局部最大值，将非极大值点
所对应的灰度值置为0，这样可以剔除掉一大部分非边缘的点。
5. 用双阈值算法检测和连接边缘

"""

"""
高斯核数值符合高斯分布
3σ准则　　在正态分布中σ代表标准差,μ代表均值x=μ即为图像的对称轴
　　三σ原则即为
　　数值分布在（μ—σ,μ+σ)中的概率为0.6826
　　数值分布在（μ—2σ,μ+2σ)中的概率为0.9544
　　数值分布在（μ—3σ,μ+3σ)中的概率为0.9974
　　可以认为,Y 的取值几乎全部集中在（μ—3σ,μ+3σ)]区间
　　内,超出这个范围的可能性仅占不到0.3%.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

if __name__ == '__main__':
    img = cv2.imread("lenna.png", 0)  # opencv读取灰度图片
    # cv2.imshow("Gray Image", img)
    # cv2.waitKey(0)
    # exit()
    pic_path = 'lenna.png'

    # 1、高斯平滑
    # sigma = 1.52  # 高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    # 一般有个经验的东西就是窗宽和sigma直接的关系就是窗宽等于2*3sigma+1
    dim = int(np.round(6 * sigma + 1))  # round是四舍五入函数，根据标准差求高斯核是几乘几的，也就是维度
    if dim % 2 == 0:  # 最好是奇数,不是的话加一
        dim += 1
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列
    n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核
    n2 = -1 / (2 * sigma ** 2)

    # 高斯卷积核公G(x,y)=n1*exp(n2*(x^2+y^2),（x,y)为高斯核坐标点
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    # 高斯核窗口归一化，保证图像保持原有亮度
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    dx, dy = img.shape
    img_new = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补

    # 高斯核与padding后图像进行卷积得到高斯滤波图像
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')

    # 2、求梯度。以下两个是滤波求梯度用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
    img_tidu_y = np.zeros(img_new.shape)
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)  # 最终方向
    img_tidu_x[img_tidu_x == 0] = 0.00000001  # 防止分母为0
    angle = img_tidu_y / img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')  # sobel边缘检测
    plt.axis('off')

    # 3、非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2:  # 是最大值
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    # 大于高阈值为强边缘，小于低阈值不是边缘。介于中间是弱边缘
    for i in range(1, dx - 1):  # 外圈不考虑了
        for j in range(1, dy - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])  # 记录强边缘像素坐标
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0

    # 弱边缘像素及其8个邻域像素，只要其中一个为强边缘像素，则该弱边缘点就可以保留为真实的边缘
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        print(len(zhan))
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if i == j == 1:
                    continue
                else:
                    if (a[i, j] < high_boundary) and (a[i, j] > lower_boundary):
                        img_yizhi[i + temp_1 - 1, j + temp_2 - 1] = 255  # 这个像素点标记为边缘
                        zhan.append([i + temp_1 - 1, j + temp_2 - 1])  # 进栈,确定为边缘点后，变更为强边缘

    # 将非边缘点置0
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0

    # 绘图
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()

    # plt.figure(5)
    # lower_boundary = img_tidu.mean() * 0.5
    # high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    # plt.imshow((cv2.Canny(img, lower_boundary, high_boundary)).astype(np.uint8), cmap='gray')
    # plt.axis('off')  # 关闭坐标刻度值
    # plt.show()
