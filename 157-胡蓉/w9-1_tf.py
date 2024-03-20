# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w9-1_tf.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略警告

# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # 随机生成[-0.5,0.5]之间的200个数据(200,)，并增加1个列的维度，即shape=（200,1）
noise = np.random.normal(0, 0.01, x_data.shape)  # 生成大小为x_data.shape，均值为0，标准差为0.02的正态分布数据作为噪音
y_data = np.square(x_data) + noise
# y_data = np.square(x_data)

# 定义两个placeholder存放输入数据
x = tf.compat.v1.placeholder(tf.float32, [None, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random.normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))  # 加入偏置项
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1  # tf.matmul(a,b)将矩阵 a 乘以矩阵 b,生成a * b
L1 = tf.nn.tanh(Wx_plus_b_L1)  # 加入激活函数

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random.normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))  # 加入偏置项
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)  # 加入激活函数

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - prediction))
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.compat.v1.Session() as sess:
    # 变量初始化
    sess.run(tf.compat.v1.global_variables_initializer())
    # 训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值,lw：折线图的线条宽度
    plt.show()
