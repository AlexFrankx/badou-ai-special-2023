from __future__ import division
from nets.frcnn import get_frcnn_train_model
from nets.frcnn_training import cls_loss, smooth_l1, Generator, get_img_output_length, class_loss_cls, class_loss_regr

from utils.config import Config
from utils.utils import BBoxUtility
from utils.roi_helpers import calc_iou
from utils.anchors import get_anchors

from keras.utils import generic_utils
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import keras
import numpy as np
import time

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略警告
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # 设置日志级别


def tfversion():
    tfversion=tf.__version__
    return int(tfversion.split(".")[0])


TFVS = tfversion()

if TFVS == 1:
    def write_log(callback, names, logs, batch_no):
        for i, name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()
else:
    def write_log(callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            with callback.as_default():
                tf.summary.scalar(name, value, step=batch_no)
                callback.flush()

if __name__ == "__main__":
    if TFVS == 2:  # tensorflow2.0 gpu版
        '''前期工作-设置GPU（如果使用的是CPU可以忽略这步）'''
        # 检查GPU是否可用
        print(tf.test.is_built_with_cuda())
        gpus = tf.config.list_physical_devices("GPU")
        print(gpus)
        if gpus:
            gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
            tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存用量按需使用
            tf.config.set_visible_devices([gpu0], "GPU")

    config = Config()  # 一些通用参数配置
    NUM_CLASSES = 21  # 分类数量+1(置信度）
    EPOCH = 2  # 训练代数100
    EPOCH_LENGTH = 200  # 一代的训练次数2000

    # =======================数据加载=======================
    # 解析图像标注文件，获取样本相关信息
    annotation_path = '2007_train.txt'  # 图像的标注文件
    with open(annotation_path) as f:
        lines = f.readlines()

    bbox_util = BBoxUtility(overlap_threshold=config.rpn_max_overlap,
                            ignore_threshold=config.rpn_min_overlap)  # 实例化bbox_util，指定roi的上限和下线阈值（rpn网络）

    gen = Generator(bbox_util, lines, NUM_CLASSES, solid=True)  # 实例化Generator
    rpn_train = gen.generate()  # gen.generate，用以生成样本Tensor

    # =======================网络模型构建=======================
    # 创建FasterRnn网络模型
    model_rpn, model_classifier, model_all = get_frcnn_train_model(config, NUM_CLASSES)

    # line_length :打印模型的字符总宽度，positions：四列分割线所在的宽度百分比位置，增加宽度可让未显示出来的字符显现
    model_rpn.summary(line_length=120, positions=[.35, .66, .78, 1.])
    model_classifier.summary(line_length=120, positions=[.35, .66, .78, 1.])
    model_all.summary(line_length=120, positions=[.35, .66, .78, 1.])

    # 加载预训练权重
    base_net_weights = "model_data/voc_weights.h5"
    model_rpn.load_weights(base_net_weights, by_name=True)
    model_classifier.load_weights(base_net_weights, by_name=True)


    # 可以使用Tensorboard --logdir=logs的绝对路径查看网路情况
    log_dir = "logs"
    if TFVS == 1:  # Tensorflow1.x
        logging = TensorBoard(log_dir=log_dir)
        callback = logging
        callback.set_model(model_all)
    else:  # Tensorflow2.x
        callback = tf.summary.create_file_writer(log_dir)

    # =======================模型编译=======================
    # 训练参数设置
    model_rpn.compile(loss={
        'regression': smooth_l1(),  # 计算regression层的loss函数，采用smooth_l1
        'classification': cls_loss()  # 计算classification层的loss函数
    }, optimizer=keras.optimizers.Adam(lr=1e-5)
    )
    model_classifier.compile(loss=[
        class_loss_cls,
        class_loss_regr(NUM_CLASSES - 1)
    ],
        metrics={'dense_class_{}'.format(NUM_CLASSES): 'accuracy'}, optimizer=keras.optimizers.Adam(lr=1e-5)
    )
    model_all.compile(optimizer='sgd', loss='mae')

    # =======================模型训练=======================
    # 初始化参数
    iter_num = 0  # 训练次数记录变量
    train_step = 0  # 与iter_num的区别？
    losses = np.zeros((EPOCH_LENGTH, 5))  # 存放loss与acc的数组，包括两个rpn的loss,两个检测模块的loss,一个检测模块的acc
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()
    best_loss = np.Inf  # 最佳loss

    # 开始训练
    print('Starting training')
    for i in range(EPOCH):

        if i == 20:  # 当执行到20代后，降低学习率
            model_rpn.compile(loss={
                'regression': smooth_l1(),
                'classification': cls_loss()
            }, optimizer=keras.optimizers.Adam(lr=1e-6)
            )
            model_classifier.compile(loss=[
                class_loss_cls,
                class_loss_regr(NUM_CLASSES - 1)
            ],
                metrics={'dense_class_{}'.format(NUM_CLASSES): 'accuracy'}, optimizer=keras.optimizers.Adam(lr=1e-6)
            )
            print("Learning rate decrease")

        progbar = generic_utils.Progbar(EPOCH_LENGTH)  # 进度条设置，长度EPOCH_LENGTH
        print('Epoch {}/{}'.format(i + 1, EPOCH))
        while True:
            if len(rpn_accuracy_rpn_monitor) == EPOCH_LENGTH and config.verbose:  # 当完成EPOCH_LENGTH次训练后
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(
                    rpn_accuracy_rpn_monitor)  # 计算roi的平均值
                rpn_accuracy_rpn_monitor = []  # 清空rpn_accuracy_rpn_monitor
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                    mean_overlapping_bboxes, EPOCH_LENGTH))
                if mean_overlapping_bboxes == 0:  # 未检测到与目标框重叠的的框
                    print(
                        'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            """RPN网络训练与预测"""
            X, Y, boxes = next(rpn_train)  # 获取训练样本数据（X图片，Y正负样本类别，预测框的变化量(12996个先验框），boxes目标框及其目标类别
            loss_rpn = model_rpn.train_on_batch(X, Y)  # 训练得出当前轮次的loss, 一次喂一张图
            write_log(callback, ['rpn_cls_loss', 'rpn_reg_loss'], loss_rpn, train_step)
            P_rpn = model_rpn.predict_on_batch(X)  # 通过训练出来的权重值预测rpn模块的预测值，sigmod实现二分类，分类值显示的是0~1之间的数

            """RPN后的分类网络训练与预测-输入数据准备：featuremap(数据归一化),建议框（300个）类别，建议框变化量"""
            height, width, _ = np.shape(X[0])  # 获取输入样本图尺寸
            # 获取原图中的先验框，坐标已做归一化处理，由每个特征点有9个先验框还原到原图中的坐标而来,如果输入样本尺寸一样，则先验框位置均一样
            anchors = get_anchors(get_img_output_length(width, height), width, height)
            # 将预测结果进行解码，得到按置信度从大到小排序的前300个框
            #  rpn预测值（△x,△y,△w,△h)-->建议框（gx,gy,gw,h)-->取置信度大于0.5的框-->每类非极大值抑制取前300个框
            #  -->所有框按置信度从大到小排序取300个
            results = bbox_util.detection_out(P_rpn, anchors, 1, confidence_threshold=0)  # 针对正样本
            R = results[0][:, 2:]
            # X2：iou>=0.1的建议框，Y1:iou>=0.1的建议框对应类别（one-hot)，Y2：iou>=0.1的建议框对应的变化量及类别，IouS：每个建议框的iou值
            X2, Y1, Y2, IouS = calc_iou(R, config, boxes[0], width, height, NUM_CLASSES)

            if X2 is None:  # 没有匹配的建议框
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, -1] == 0)  # 获取负样本的索引号
            pos_samples = np.where(Y1[0, :, -1] == 1)  # 获取正样本的索引号

            if len(neg_samples) > 0:  # tuple转list
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if len(neg_samples) == 0:
                continue

            if len(pos_samples) < config.num_rois // 2:
                selected_pos_samples = pos_samples.tolist()
            else:
                # 在pos_samples中随机抽取不重复的config.num_rois//2个数据，即众多框中一共只取config.num_rois个数据，正负样本各占一半
                selected_pos_samples = np.random.choice(pos_samples, config.num_rois // 2, replace=False).tolist()
            try:
                # 负样本数量大于config.num_rois//2时，不重复抽取，否则支持重复抽取
                selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples),
                                                        replace=False).tolist()
            except:
                selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples),
                                                        replace=True).tolist()

            # 正负样本拼接在一起合为总样本，总共32个sel_samples建议框的索引号
            sel_samples = selected_pos_samples + selected_neg_samples
            # model_classifier为多输出模型既有loss，也有metrics, 此时 loss_class 为一个列表，
            # 代表这个 mini-batch 的 两个loss 和 metrics
            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                         [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            write_log(callback, ['detection_cls_loss', 'detection_reg_loss', 'detection_acc'], loss_class, train_step)

            losses[iter_num, 0] = loss_rpn[1]  # rpn_cls_loss
            losses[iter_num, 1] = loss_rpn[2]  # rpn_reg_loss
            losses[iter_num, 2] = loss_class[1]  # detection_cls_loss 顺序怎么排列的，是按模型返回的顺序
            losses[iter_num, 3] = loss_class[2]  # detection_reg_loss
            losses[iter_num, 4] = loss_class[3]  # detection_acc

            train_step += 1
            iter_num += 1

            # 更新进度条
            progbar.update(iter_num,
                           [('rpn_cls', np.mean(losses[:iter_num, 0])),
                            ('rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('detector_cls', np.mean(losses[:iter_num, 2])),
                            ('detector_regr', np.mean(losses[:iter_num, 3])),
                            ('detection_acc', np.mean(losses[:iter_num, 4]))])

            if iter_num == EPOCH_LENGTH:  # 运行到最大的一轮
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if config.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                write_log(callback,
                          ['Elapsed_time', 'mean_overlapping_bboxes', 'mean_rpn_cls_loss', 'mean_rpn_reg_loss',
                           'mean_detection_cls_loss', 'mean_detection_reg_loss', 'mean_detection_acc', 'total_loss'],
                          [time.time() - start_time, mean_overlapping_bboxes, loss_rpn_cls, loss_rpn_regr,
                           loss_class_cls, loss_class_regr, class_acc, curr_loss], i)

                if config.verbose:
                    print('The best loss is {}. The current loss is {}. Saving weights'.format(best_loss, curr_loss))
                if curr_loss < best_loss:
                    best_loss = curr_loss
                model_all.save_weights(log_dir + "/epoch{:03d}-loss{:.3f}-rpn{:.3f}-roi{:.3f}".format(i, curr_loss,
                                                                                                      loss_rpn_cls + loss_rpn_regr,
                                                                                                      loss_class_cls + loss_class_regr) + ".h5")

                break
