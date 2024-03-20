# -------------------------------------------------------------#
#   InceptionV3的网络部分
# -------------------------------------------------------------#

# 低版本的Python2.X在使用print的时候也需要用Python3.X的语法
from __future__ import print_function
# 当当前库与标准库出现相同库，import导入时，如果需要导入当前库，则需要加上当前文件夹路径如from currentdir import xxx
from __future__ import absolute_import

import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略警告
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # 设置日志级别


def conv2d_bn(x, filters, num_row, num_col, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False, name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x





def InceptionModuleApart(x, pool_conv_oc, name):
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, pool_conv_oc, 1, 1)

    # oc=64+64+96+pool_conv_oc    nhwc-0123
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name=name)
    return x

def InceptionModuleBpart1(x, name):
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3, name=name)
    return x


def InceptionModuleBpart(x, conv2d_bn_temp_oc, name):
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, conv2d_bn_temp_oc, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, conv2d_bn_temp_oc, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, conv2d_bn_temp_oc, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, conv2d_bn_temp_oc, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, conv2d_bn_temp_oc, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, conv2d_bn_temp_oc, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name=name)
    return x


def InceptionModuleCpart1(x, name):
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name=name)
    return x


def InceptionModuleCpart(x, name):
    branch1x1 = conv2d_bn(x, 320, 1, 1)

    branch3x3 = conv2d_bn(x, 384, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
    branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3)

    branch3x3dbl = conv2d_bn(x, 448, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3, name=name)
    return x


def InceptionModuleA(x):
    # part1:branch1x1, branch5x5, branch3x3dbl, branch_pool
    x = InceptionModuleApart(x, 32, 'mixed0')  # 35 x 35 x 192 -> 35 x 35 x 256, oc = 64 + 64 + 96 + 32 = 256

    # part2：branch1x1, branch5x5, branch3x3dbl, branch_pool
    x = InceptionModuleApart(x, 64, 'mixed1')  # 35 x 35 x 256 -> 35 x 35 x 288, oc = 64 + 64 + 96 + 64 = 288

    # part3：branch1x1, branch5x5, branch3x3dbl, branch_pool
    x = InceptionModuleApart(x, 64, 'mixed2')  # 35 x 35 x 256 -> 35 x 35 x 288, oc = 64 + 64 + 96 + 64 = 288

    return x


def InceptionModuleB(x):
    # part1:branch3x3, branch3x3dbl, branch_pool
    x = InceptionModuleBpart1(x, "mixed3")  # 35 x 35 x 288 -> 17 x 17 x 768, oc = 384 + 64 + 288 = 768

    # part2:branch1x1, branch7x7, branch7x7dbl, branch_pool
    x = InceptionModuleBpart(x, 128, "mixed4")  # 17 x 17 x 768 -> 17 x 17 x 768, oc = 192 + 192 + 192 + 192 = 768

    # part3:branch1x1, branch7x7, branch7x7dbl, branch_pool
    x = InceptionModuleBpart(x, 160, "mixed5")  # 17 x 17 x 768 -> 17 x 17 x 768, oc = 192 + 192 + 192 + 192 = 768

    # part4:branch1x1, branch7x7, branch7x7dbl, branch_pool
    x = InceptionModuleBpart(x, 160, "mixed6")  # 17 x 17 x 768 -> 17 x 17 x 768, oc = 192 + 192 + 192 + 192 = 768

    # part5:branch1x1, branch7x7, branch7x7dbl, branch_pool
    x = InceptionModuleBpart(x, 192, "mixed7")  # 17 x 17 x 768 -> 17 x 17 x 768, oc = 192 + 192 + 192 + 192 = 768

    return x


def InceptionModuleC(x):
    # part1：branch3x3, branch7x7x3, branch_pool
    x = InceptionModuleCpart1(x, "mixed8")  # 17 x 17 x 768 -> 8 x 8 x 1280, oc = 768 + 320 + 192  = 1280

    # part2:branch1x1, branch3x3(branch3x3_1, branch3x3_2), branch3x3dbl(branch3x3dbl_1, branch3x3dbl_2), branch_pool
    x = InceptionModuleCpart(x, "mixed9")  # 8 x 8 x 1280 -> 8 x 8 x 2048

    # part3:branch1x1, branch3x3(branch3x3_1, branch3x3_2), branch3x3dbl(branch3x3dbl_1, branch3x3dbl_2), branch_pool
    x = InceptionModuleCpart(x, "mixed10")  # 8 x 8 x 2048 -> 8 x 8 x 2048
    return x


# 仅实现主干，未实现辅助分支
def InceptionV3(input_shape=[299, 299, 3], classes=1000):
    img_input = Input(shape=input_shape)

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # os = 35 x 35 x 192

    # part1：branch1x1, branch5x5, branch3x3dbl, branch_pool
    # part2：branch1x1, branch5x5, branch3x3dbl, branch_pool
    # part3：branch1x1, branch5x5, branch3x3dbl, branch_pool
    x = InceptionModuleA(x)  # os = 35 x 35 x 288

    # part1:branch3x3, branch3x3dbl, branch_pool
    # part2:branch1x1, branch7x7, branch7x7dbl, branch_pool
    # part3:branch1x1, branch7x7, branch7x7dbl, branch_pool
    # part4:branch1x1, branch7x7, branch7x7dbl, branch_pool
    # part5:branch1x1, branch7x7, branch7x7dbl, branch_pool
    x = InceptionModuleB(x)  # os = 17 x 17 x 768

    # part1：branch3x3, branch7x7x3, branch_pool
    # part2:branch1x1, branch3x3(branch3x3_1, branch3x3_2), branch3x3dbl(branch3x3dbl_1, branch3x3dbl_2), branch_pool
    # part3:branch1x1, branch3x3(branch3x3_1, branch3x3_2), branch3x3dbl(branch3x3dbl_1, branch3x3dbl_2), branch_pool
    x = InceptionModuleC(x)  # os = 8 x 8 x 2048

    # 平均池化后全连接。
    x = GlobalAveragePooling2D(name='avg_pool')(x)  # os = 2048
    x = Dense(classes, activation='softmax', name='predictions')(x)  # os = 1000

    inputs = img_input

    model = Model(inputs, x, name='inception_v3')
    model.summary()
    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = InceptionV3()

    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
