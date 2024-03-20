# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w3-2_GaussianNoise.py


import numpy as np
import cv2
import random
from skimage import util

def GaussianNoise(src, means, sigma, percetage):
    """随机生成符合正态（高斯）分布的随机数，means,sigma为两个参数",src为单通道图片"""
    # 克隆图像
    NoiseImg=np.copy(src)
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 高斯噪声图片边缘不处理，故-1
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)

        # 此处在原有像素灰度值上加上满足高斯分布的随机数
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)

        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


img = cv2.imread('lenna.png')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 调用自定义函数实现高斯噪声
img2 = GaussianNoise(img1, 2, 4, 0.8)

# 将灰度图与高斯噪声图拼接在一起显示
cv2.imshow('gray_GaussianNoise', np.hstack([img1, img2]))


"""
def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
功能：为浮点型图片添加各种随机噪声
参数：
image：输入图片（将会被转换成浮点型），ndarray型
mode： 可选择，str型，表示要添加的噪声类型
	gaussian：高斯噪声
	localvar：高斯分布的加性噪声，在“图像”的每个点处具有指定的局部方差。
	poisson：泊松噪声
	salt：盐噪声，随机将像素值变成1
	pepper：椒噪声，随机将像素值变成0或-1，取决于矩阵的值是否带符号
	s&p：椒盐噪声
	speckle：均匀噪声（均值mean方差variance），out=image+n*image
seed： 可选的，int型，如果选择的话，在生成噪声前会先设置随机种子以避免伪随机
clip： 可选的，bool型，如果是True，在添加均值，泊松以及高斯噪声后，会将图片的数据裁剪到合适范围内。如果谁False，则输出矩阵的值可能会超出[-1,1]
mean： 可选的，float型，高斯噪声和均值噪声中的mean参数，默认值=0
var：  可选的，float型，高斯噪声和均值噪声中的方差，默认值=0.01（注：不是标准差）
local_vars：可选的，ndarry型，用于定义每个像素点的局部方差，在localvar中使用
amount： 可选的，float型，是椒盐噪声所占比例，默认值=0.05
salt_vs_pepper：可选的，float型，椒盐噪声中椒盐比例，值越大表示盐噪声越多，默认值=0.5，即椒盐等量
--------
返回值：ndarry型，且值在[0,1]或者[-1,1]之间，取决于是否是有符号数
"""

def min_max_normalize(x):
    """图像归一化，并转换倒0-255之间"""
    x_min = np.min(x)
    x_max = np.max(x)
    x_nor=(x - x_min) / (x_max - x_min)
    r_img=(x_nor * 255).astype(np.uint8)
    return r_img

# 直接调用接口对原图实现高斯噪声
noise_gs_img = util.random_noise(img, mode='gaussian')
img = cv2.imread('lenna.png')
bgr_img = img[:, :, ::-1]
img_show=min_max_normalize(img)
noise_gs_img_show=min_max_normalize(noise_gs_img)

# 将原图与高斯噪声图拼接在一起显示
cv2.imshow('bgr_GaussianNoise',  np.hstack([img_show, noise_gs_img_show]))
cv2.waitKey(0)
