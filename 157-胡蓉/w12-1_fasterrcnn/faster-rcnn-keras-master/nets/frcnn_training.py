from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
import keras
import tensorflow as tf
import numpy as np
from random import shuffle
import random
from PIL import Image
# from keras.objectives import categorical_crossentropy  # 高版本keras不支持注释掉，改为下一行即可。
from keras.losses import categorical_crossentropy
from keras.utils.data_utils import get_file
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from utils.anchors import get_anchors
import time


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def cls_loss(ratio=3):
    def _cls_loss(y_true, y_pred):
        # y_true [batch_size, num_anchor, num_classes+1]
        # y_pred [batch_size, num_anchor, num_classes]
        labels = y_true
        anchor_state = y_true[:, :, -1]  # -1 是需要忽略的, 0 是背景, 1 是存在目标
        classification = y_pred

        # 找出存在目标的先验框
        indices_for_object = tf.where(keras.backend.equal(anchor_state, 1))
        labels_for_object = tf.gather_nd(labels, indices_for_object)
        classification_for_object = tf.gather_nd(classification, indices_for_object)

        cls_loss_for_object = keras.backend.binary_crossentropy(labels_for_object, classification_for_object)

        # 找出实际上为背景的先验框
        indices_for_back = tf.where(keras.backend.equal(anchor_state, 0))
        labels_for_back = tf.gather_nd(labels, indices_for_back)
        classification_for_back = tf.gather_nd(classification, indices_for_back)

        # 计算每一个先验框应该有的权重
        cls_loss_for_back = keras.backend.binary_crossentropy(labels_for_back, classification_for_back)

        # 标准化，实际上是正样本的数量
        normalizer_pos = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer_pos = keras.backend.cast(keras.backend.shape(normalizer_pos)[0], keras.backend.floatx())
        normalizer_pos = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer_pos)

        normalizer_neg = tf.where(keras.backend.equal(anchor_state, 0))
        normalizer_neg = keras.backend.cast(keras.backend.shape(normalizer_neg)[0], keras.backend.floatx())
        normalizer_neg = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer_neg)

        # 将所获得的loss除上正样本的数量
        cls_loss_for_object = keras.backend.sum(cls_loss_for_object) / normalizer_pos
        cls_loss_for_back = ratio * keras.backend.sum(cls_loss_for_back) / normalizer_neg

        # 总的loss，给正负样本都给与一个权重系数，这样就避免正样本少，
        # 模型就倾向于将所有的值都预测为0，这样损失也是下降的，精度也会提升，但是正样本无法被‘重视’，尤其是对于要预测正样本的任务，就没有意义了
        loss = cls_loss_for_object + cls_loss_for_back

        return loss

    return _cls_loss


def smooth_l1(sigma=1.0):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # y_true [batch_size, num_anchor, 4+1]
        # y_pred [batch_size, num_anchor, 4]
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # 找到正样本
        """
        tf.where()返回一个布尔张量中真值的位置。对于非布尔型张量，非0的元素都判为True
        返回的是二维张量，第一个维度的数量，即行数表明有多少个为True值；每一行中的数据代表该True值在原张量中的位置。
        """
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)  # 取出索引所在位置的数据，即正样本的anchor
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算 smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)  # 计算元素绝对值
        """
        where(condition, x=None, y=None, name=None)的用法
        condition， x, y 相同维度，condition是bool型值，True/False        
        返回值是对应元素，condition中元素为True的元素替换为x中的元素，为False的元素替换为y中对应元素
        
        对于大多数CNN网络，我们一般是使用L2-loss而不是L1-loss，因为L2-loss的收敛速度要比L1-loss要快得多。
        对于边框预测回归问题，通常也可以选择平方损失函数（L2损失），但L2范数的缺点是当存在离群点（outliers)的时候，
        这些点会占loss的主要组成部分。比如说真实值为1，预测10次，有一次预测值为1000，其余次的预测值为1左右，
        显然loss值主要由1000主宰。所以FastRCNN采用稍微缓和一点绝对损失函数（smooth L1损失），
        它是随着误差线性增长，而不是平方增长。
        smooth L1和L1-loss函数的区别在于，L1-loss在0点处导数不唯一，可能影响收敛。
        smooth L1的解决办法是在0点附近使用平方函数使得它更加平滑。
        smooth L1公式：当|x|<1时，0.5*x^2,否则|x|-0.5
        """
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),  # 逐个比对x<=y的真值，返回布尔量
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])  #
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss = keras.backend.sum(regression_loss) / normalizer  # 所有正样本的loss求平均

        return loss

    return _smooth_l1


def class_loss_regr(num_classes):
    epsilon = 1e-4

    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4 * num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        loss = 4 * K.sum(
            y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
            epsilon + y_true[:, :, :4 * num_classes])  # label标签与目标框同等维度，便于计算改框所在label的loss
        return loss

    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))  # 分类交叉熵求均值


def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height


def get_img_output_length(width, height): # 原图尺寸，经过cnn后的尺寸
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [7, 3, 1, 1]
        padding = [3, 1, 0, 0]
        stride = 2
        for i in range(4):
            # input_length = (input_length - filter_size + stride) // stride
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length

    return get_output_length(width), get_output_length(height)


class Generator(object):
    """
    生成RPN样本集
    bbox_util：先验框对象
    num_classes：类别数量
    solid：是否采用固定尺寸
    solid_shape：固定尺寸
    """
    def __init__(self, bbox_util,
                 train_lines, num_classes, solid, solid_shape=[600, 600]):
        self.bbox_util = bbox_util
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.num_classes = num_classes
        self.solid = solid
        self.solid_shape = solid_shape

    def get_random_data(self, annotation_line, jitter=.1, hue=.1, sat=1.1, val=1.1,):
        '''实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        if self.solid:  # 固定缩放尺寸
            w, h = self.solid_shape
        else:  # 根据图像的窄边缩放尺寸
            w, h = get_new_img_size(iw, ih)
        # 获取样本的框坐标及类别（x1,y1,x2,y2,cls),将字符串列表转换为二维数组
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # resize image 宽高乘以随机缩放系数得到新的宽高
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(.9, 1.1)

        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)  # nw < nh
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)  # nw >= nh
        image = image.resize((nw, nh), Image.BICUBIC)  # 双三次插值

        # place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        # 将image粘贴到new_image时，(dx,dy)为image的左上角坐标，
        # 如果dx,dy为负，表示image裁剪了这么多之后然后以(0,0)为左上角坐标粘贴。
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand() < .5  # 随机翻转
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)  # 将图像转到HSV颜色空间
        # 随机更改h s v 值
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image_data = hsv_to_rgb(x) * 255  # numpy array, 0 to 1，将图像还原到RGB空间

        # correct boxes 根据图像的缩放翻转裁剪等信息修正框的位置
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box,舍弃点box_w和box_h<=1的无效框
            box_data[:len(box)] = box  # 因为图像缩放裁剪等
        if len(box) == 0:  # 没有标注框则不用修正框，直接返回
            return image_data, []

        if (box_data[:, :4] > 0).any():  # 修正后的存在有效框则返回，包含被舍弃的框全0
            return image_data, box_data
        else:
            return image_data, []

    def generate(self):
        while True:
            shuffle(self.train_lines)
            lines = self.train_lines
            batchdata_X = []
            batchdata_Y = []
            batchdata_boxes = []
            num=0
            for annotation_line in lines:
                # 为输入图像做随机图像增强
                img, y = self.get_random_data(annotation_line)
                height, width, _ = np.shape(img)

                if len(y) == 0:  # 如果没有有效框，则换下一张图
                    continue
                boxes = np.array(y[:, :4], dtype=np.float32)  # 图像增强后的目标框(x1,y1,x2,y2)
                boxes[:, 0] = boxes[:, 0] / width  # 归一化
                boxes[:, 1] = boxes[:, 1] / height
                boxes[:, 2] = boxes[:, 2] / width
                boxes[:, 3] = boxes[:, 3] / height

                box_heights = boxes[:, 3] - boxes[:, 1]
                box_widths = boxes[:, 2] - boxes[:, 0]
                if (box_heights <= 0).any() or (box_widths <= 0).any():
                    continue  # 无效框则忽略

                y[:, :4] = boxes[:, :4]  # 最终有效的目标框

                # 生成固定大小的9个先验框，每个特征点还原到原图中的坐标，每个还原点作为先验框的左上点，得出每个点的先验框，再对先验框在原图坐标下做归一化处理
                anchors = get_anchors(get_img_output_length(width, height), width, height)

                # 计算真实框对应的先验框，与这个先验框应当有的预测结果(△x,△x,△w,△h,isobj)，传入参数均为归一化数据
                assignment = self.bbox_util.assign_boxes(y, anchors)

                num_regions = 256

                classification = assignment[:, 4]
                regression = assignment[:, :]

                mask_pos = classification[:] > 0 #正样本为1
                num_pos = len(classification[mask_pos]) #正样本个数
                if num_pos > num_regions / 2:
                    val_locs = random.sample(range(num_pos), int(num_pos - num_regions / 2))
                    classification[mask_pos][val_locs] = -1
                    regression[mask_pos][val_locs, -1] = -1

                mask_neg = classification[:] == 0  # 负样本为0
                num_neg = len(classification[mask_neg])  # 负样本个数
                if len(classification[mask_neg]) + num_pos > num_regions:  # 当正负样本总数大于给定值256
                    val_locs = random.sample(range(num_neg), int(num_neg - num_pos))  # 在负样本的个数范围内，随机取负样本大于正样本的数量的值
                    classification[mask_neg][val_locs] = -1  # 把超出的那部分负样本框置为忽略框，保证正负样本数量一致

                classification = np.reshape(classification, [-1, 1])
                regression = np.reshape(regression, [-1, 5])

                tmp_inp = np.array(img)
                # 分类预测值(正样本，负样本，忽略样本）+回归的变化量预测值
                tmp_targets = [np.expand_dims(np.array(classification, dtype=np.float32), 0),
                               np.expand_dims(np.array(regression, dtype=np.float32), 0)]
                # np.expand_dims(tmp_inp, 0)：图像numpy数组，
                # tmp_targets：预测值（分类预测：正负样本判断+回归（变化量）预测，
                # np.expand_dims(y, 0)：目标框坐标+分类结果
                # preprocess_input()：这是tensorflow下keras自带的类似于一个归一化的函数
                # 实现了不同传入数据格式及模式判断，并进行相应处理的；可以看到默认传入的模式tf对应的执行操作如上：
                # 即在原有传入图片数组值(0-255)的基础之上，进行先除以 /127.5，然后减1，最后得到值得范围为(-1,1)

                yield preprocess_input(np.expand_dims(tmp_inp, 0)), tmp_targets, np.expand_dims(y, 0)
