# -*- coding: utf-8 -*-
# @Author : Ruby
# @File : w8-1_keras_minist.py

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


# 加载手写体数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)


# 将数据集前20个图片数据可视化显示
# 进行图像大小为20宽、10长的绘图(单位为英寸inch)
plt.figure(figsize=(20, 10))
# 遍历MNIST数据集下标数值0~49
for i in range(20):
    # 将整个figure分成4行5列，绘制第i+1个子图。
    plt.subplot(4, 5, i + 1)
    # 设置不显示x轴刻度
    plt.xticks([])
    # 设置不显示y轴刻度
    plt.yticks([])
    # 设置不显示子图网格线
    plt.grid(False)
    # 图像展示，cmap为颜色图谱，"plt.cm.binary"为matplotlib.cm中的色表
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # 设置x轴标签显示为图片对应的数字
    plt.xlabel(train_labels[i])
# 显示图片
plt.show()


# # 数据归一化
train_images = train_images.reshape((60000, 28*28))
test_images = test_images.reshape((10000, 28*28))
train_images, test_images = train_images / 255.0, test_images / 255.0

# 对标签进行one-hot编码
print("before change:", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])


# 构建神经网络
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Flatten())
network.add(layers.Dense(10, activation='softmax'))


# 编译
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])

# 训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 数据评估
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

# 数据预测
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[8]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = digit.reshape((1, 28*28))
res = network.predict(test_images)

for i in range(res[0].shape[0]):
    if (res[0][i] == 1):
        print("the number for the picture is : ", i)
        break