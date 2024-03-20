import numpy as np
import keras
import tensorflow as tf
from utils.config import Config
import matplotlib.pyplot as plt

config = Config()

def generate_anchors(sizes=None, ratios=None):
    if sizes is None:
        sizes = config.anchor_box_scales  # [128,256,512]

    if ratios is None:
        ratios = config.anchor_box_ratios  # [[1,1],[1,2],[2,1]]

    num_anchors = len(sizes) * len(ratios)  # anchor数量

    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T
    
    for i in range(len(ratios)):
        anchors[3*i:3*i+3, 2] = anchors[3*i:3*i+3, 2]*ratios[i][0]
        anchors[3*i:3*i+3, 3] = anchors[3*i:3*i+3, 3]*ratios[i][1]
    

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors

def shift(shape, anchors, stride=config.rpn_stride):
    """
    特征图上的每个店还原到原图后，每个点9个先验框
    """
    # keras.backend.floatx() 该函数以字符串形式返回默认的float类型（float32）
    # np.arange([start, ]stop, [step, ]dtype=None) 默认步长1
    # 将每一个点还原到原图中的点坐标，输入shape=(38,38)
    shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride  # (38,)
    shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride  # (38,)

    # X, Y = np.meshgrid(x, y) 代表的是将x中每一个数据和y中每一个数据组合生成很多点,
    # 然后将这些点的x坐标放入到X中,y坐标放入Y中,并且相应位置是对应的
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # (38,38)

    shift_x = np.reshape(shift_x, [-1])  # (1444,)
    shift_y = np.reshape(shift_y, [-1])

    shifts = np.stack([shift_x, shift_y, shift_x, shift_y], axis=0)  # (4,1444)

    shifts            = np.transpose(shifts)  # (1444,4)
    number_of_anchors = np.shape(anchors)[0]  # 9

    k = np.shape(shifts)[0]  # 1444

    # 默认框的9个anchor框，原shape=(9,4),reshape为(1,9,4)
    # 由特征图还原到原图的中心点原shape=(1444,4),reshape为(1444,1,4)
    # 形状不同的数组相加自动触发广播机制，相加后shape为(1444,9,4)
    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    return shifted_anchors


def get_anchors(shape,width,height):
    "在原图中生成归一化处理后的先验框"
    anchors = generate_anchors()  # 生成9个anchors
    network_anchors = shift(shape, anchors)  # 特征图上的每个点还原到原图后的所有anchors,shape=(12996,4)
    network_anchors[:, 0] = network_anchors[:, 0] / width  # 将目标框归一化到0,1之间
    network_anchors[:, 1] = network_anchors[:, 1] / height
    network_anchors[:, 2] = network_anchors[:, 2] / width
    network_anchors[:, 3] = network_anchors[:, 3] / height
    network_anchors = np.clip(network_anchors, 0, 1)  # 将结果小于0数据赋值0,大于1的数据赋值1，即更改超界框坐标
    return network_anchors
