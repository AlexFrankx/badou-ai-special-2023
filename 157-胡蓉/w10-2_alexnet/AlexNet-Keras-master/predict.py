import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # 设置日志级别
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略警告
K.set_image_data_format('channels_last')
if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("./logs/last1.h5")
    # img = utils.load_image("./Test.jpg")
    img = cv2.imread("./Test.jpg")
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 颜色空间转换
    img_nor = img_RGB / 255  # 归一化
    img_nor = np.expand_dims(img_nor, axis=0)  # 扩充维度（1，h,w,c)
    img_resize = utils.resize_image(img_nor, (224, 224)) # 缩放（1，224,224,c)
    # utils.print_answer(np.argmax(model.predict(img)))
    print(np.argmax(model.predict(img_resize)))
    print(utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("ooo", img)
    cv2.waitKey(0)
