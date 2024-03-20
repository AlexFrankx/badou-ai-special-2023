import numpy as np
import tensorflow as tf
from PIL import Image
import keras
import numpy as np
import math
import tensorflow.compat.v1 as tf
def tfversion():
    tfversion=tf.__version__
    return int(tfversion.split(".")[0])


TFVS = tfversion()
class BBoxUtility(object):
    def __init__(self, priors=None, overlap_threshold=0.7,ignore_threshold=0.3,
                 nms_thresh=0.7, top_k=300):
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self.ignore_threshold = ignore_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k
        if TFVS == 1:
            self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
            self.scores = tf.placeholder(dtype='float32', shape=(None,))
            # 非极大值抑制
            # 按scores由大到小排序，然后选定第一个，依次对之后的做iou ，删除那些和选定的框iou大于阈值的框，循环完第一个，
            # 再选定之后的一个，再 对它后面的框做iou，循环操作。选出最后剩余的框。选定的框不会超过设定的最大值。依次删除最小的。
            # 然后取前_top_k个框
            self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                    self._top_k,
                                                    iou_threshold=self._nms_thresh)
            self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))  # 0表示使用CPU，1则是GPU

    @property  # 只读属性不能被篡改
    def nms_thresh(self):
        return self._nms_thresh

    @nms_thresh.setter  # @*.setter 允许对已用@property装饰的函数赋值
    def nms_thresh(self, value):
        """重新设置非极大值抑制阈值并根据该阈值重新计算非极大值抑制结果"""
        self._nms_thresh = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, value):
        """重新设置非极大值抑制选取的top框数量并根据该值重新计算非极大值抑制结果"""
        self._top_k = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    def iou(self, box):
        """计算出每个真实框与所有的先验框的iou"""
        # box=(left,top,right,bottom)
        # 判断真实框与先验框的重合情况，计算交集框坐标
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])
        # 计算交集框坐标w和h，并计算交集面积
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 先验框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 真实目标框的面积
        area_gt = (self.priors[:, 2] - self.priors[:, 0])*(self.priors[:, 3] - self.priors[:, 1])
        # 计算iou
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        iou = self.iou(box)  # 计算出每个真实框与所有的先验框的iou
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))

        # 找到每一个真实框，重合程度较高的先验框，iou大于0.7
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True # 当没有iou大于0.7的，就把最大iou对应位置置为True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]  # 赋值到encoded_box中
        
        # 找到对应的先验框
        assigned_priors = self.priors[assign_mask]
        # 逆向编码，将真实框转化为FasterRCNN预测结果的格式
        # 先计算真实框的中心与长宽
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # 再计算重合度较高的先验框的中心与长宽
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])
        
        # 逆向求取FasterRCNN应该有的预测结果,求出先验框的变化量（△x,△y,△w,△h)
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] *= 4

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] *= 4
        return encoded_box.ravel()

    def ignore_box(self, box):
        """
        :param box: 真实目标框
        :return:重合程度较高的先验框对应的iou，没有重合度较高的，就取最高的，其他位置置0
        """

        iou = self.iou(box)  # 与每一个先验框都计算iou
        
        ignored_box = np.zeros((self.num_priors, 1))

        # 找到每一个真实框，重合程度较高的先验框，iou0.3到0.7之间
        assign_mask = (iou > self.ignore_threshold)&(iou<self.overlap_threshold)

        if not assign_mask.any():  # assign_mask.any()表示当任意一个不为False,结果为True
            assign_mask[iou.argmax()] = True # 当没有iou0.3到0.7之间的，就把最大iou对应位置置为True
            
        ignored_box[:, 0][assign_mask] = iou[assign_mask]  # 仅将iou是0.3~0.7的iou(没有该范围的取最大iou)保留,其余iou置0，shape=(12996,1)
        return ignored_box.ravel()  # ravel()方法将数组维度拉成一维数组,shape=(12996,)

    def assign_boxes(self, boxes, anchors):
        """
        :param boxes: 真实的目标框
        :param anchors: 先验框
        :return:先验框变化量的预测值以及对应框的是否有目标的预测值
        """
        self.num_priors = len(anchors) # 先验框数量
        self.priors = anchors  # 先验框
        assignment = np.zeros((self.num_priors, 4 + 1))

        assignment[:, 4] = 0.0
        if len(boxes) == 0:
            return assignment
            
        # 对每一个真实目标框都进行iou计算,找出重合程度较高(iou 0.3~0.7)的先验框对应的iou(没有重合度较高的，就取最高的)其他位置置0
        ingored_boxes = np.apply_along_axis(self.ignore_box, 1, boxes[:, :4])  # shape=(len(boxes),12996)
        ingored_boxes = ingored_boxes.reshape(-1, self.num_priors, 1)  # shape=(len(boxes),12996,1)
        # (num_priors)对每一个先验框取出iou最大的目标真实框与之一一对应
        ignore_iou = ingored_boxes[:, :, 0].max(axis=0)  # shape=(12996,)
        # (num_priors)取出先验框中的iou非0的框，即满足iou在0.3~0.7之间且iou最大的一个真实框
        ignore_iou_mask = ignore_iou > 0
        assignment[:, 4][ignore_iou_mask] = -1 #iou在0.3~0.7范围内的先验框置为-1，即疑似目标

        # (n, num_priors*5),求出先验框到目标框的变化量（△x,△y,△w,△h)(归一化后的值)以及对应的iou
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # 每一个真实框的编码后的值，和iou
        # (n, num_priors,5)
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        # 取重合程度最大的先验框
        # (num_priors)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        # (num_priors)重合程度最大的先验框的index，即每一个先验框的最匹配的目标框，iou>0.7范围内的
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        # (num_priors)
        best_iou_mask = best_iou > 0
        # 某个先验框它属于哪个真实框
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx) #assign_num个先验框与目标框重合度高
        # 保留重合程度最大的先验框的应该有的预测结果
        # 哪些先验框存在真实框
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]

        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,np.arange(assign_num),:4]
        # 4代表，0为背景的概率，1为目标
        assignment[:, 4][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox):
        """
        根据rpn网络输出的偏移量,移动先验框,获得建议框proposal_box
        :param mbox_loc: rpn网络输出的先验框偏移量(x,y,h,w)
        :param mbox_priorbox: 先验框(left,top,right,bottom)
        :return:proposal_box ：建议框(left,top,right,bottom)
        """

        # 获得先验框的宽与高
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]

        # 获得先验框的中心点
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        # 建议框的中心点求取，中心点(Px,Py)做平移操作变为（Gx,Gy)，偏移量(△x，△y),计算公式为Gx=Px+△x*Pw，本文中偏移量增加了缩放倍数1/4（自己设定的）
        proposal_center_x = mbox_loc[:, 0] * prior_width / 4 + prior_center_x
        proposal_center_y = mbox_loc[:, 1] * prior_height / 4 + prior_center_y

        # 建议框的宽与高的求取，宽高的偏移量（Gw,Gy)，计算公式为Gw=Pw*exp(△w),本文中偏移量增加了缩放倍数1/4
        proposal_width = np.exp(mbox_loc[:, 2] / 4) * prior_width
        proposal_height = np.exp(mbox_loc[:, 3] / 4) * prior_height

        # 获取建议框的左上角与右下角
        proposal_xmin = proposal_center_x - 0.5 * proposal_width
        proposal_ymin = proposal_center_y - 0.5 * proposal_height
        proposal_xmax = proposal_center_x + 0.5 * proposal_width
        proposal_ymax = proposal_center_y + 0.5 * proposal_height

        # 建议框的左上角与右下角进行堆叠
        # proposal_xmin[:, None]相当于proposal_xmin.reshape(-1,1)
        proposal_box = np.concatenate((proposal_xmin[:, None],
                                      proposal_ymin[:, None],
                                      proposal_xmax[:, None],
                                      proposal_ymax[:, None]), axis=-1)
        # 防止超出0与1  ，小于0取0，大于1取1
        proposal_box = np.minimum(np.maximum(proposal_box, 0.0), 1.0)
        return proposal_box

    def detection_out(self, predictions, mbox_priorbox, num_classes, keep_top_k=300,
                        confidence_threshold=0.5):
        """
        将rpn网络出来的回归预测结果转换为建议框,置信度前300的框
        rpn预测值（△x,△y,△w,△h)-->建议框（gx,gy,gw,h)-->取置信度大于0.5的框-->每类非极大值抑制取前300个框
        -->所有框按置信度从大到小排序取300个
        :param predictions: rpn网络的预测结果
        :param mbox_priorbox: 先验框/锚框 (left,top,right,bottom)(已归一化)
        :param num_classes: 样本类别（正样本负样本忽略样本）
        :param keep_top_k: 最终取的框数量
        :param confidence_threshold:
        :return:
        """

        mbox_conf = predictions[0]  # 置信度
        mbox_loc = predictions[1]  # 回归预测结果

        results = []
        # 对每一个图片进行处理
        # 训练阶段len(mbox_loc)为batch_size,预测时len(mbox_loc)为1，因为每次处理一张图片
        for i in range(len(mbox_loc)):
            results.append([])
            # 利用rpn的框变化量移动先验框,获得建议框坐标，decode_bbox就是建议框(left,top,right,bottom)(归一化的值）
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox)
            for c in range(num_classes):
                #   取出先验框内包含物体的概率
                c_confs = mbox_conf[i, :, c]

                # 可通过对所有rpn网络预测的先验框是否包含物体的概率,由大到小排序, 获得索引，取前面的指定数量个框
                # 这里采用取置信度大于0.5的框
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    # 取出得分高于confidence_threshold的框及其置信度
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]

                    # 进行iou的非极大抑制
                    if TFVS == 1:
                        feed_dict = {self.boxes: boxes_to_process,
                                        self.scores: confs_to_process}
                        idx = self.sess.run(self.nms, feed_dict=feed_dict)
                    else:
                        # self.nms为非极大值抑制函数，idx为进行非极大值抑制后框的索引，这里设定默认取300个框
                        idx = tf.image.non_max_suppression(boxes_to_process, confs_to_process,
                                                            self._top_k,
                                                            iou_threshold=self._nms_thresh)

                    # 取出在非极大抑制中效果较好的内容
                    good_boxes = boxes_to_process[idx]  # shape=(300,4)
                    confs = confs_to_process[idx][:, None]  # shape=(300,1)
                    # 将label、置信度、框的位置进行堆叠。
                    labels = c * np.ones((len(idx), 1))   # shape=(300,1),内容300个c
                    c_pred = np.concatenate((labels, confs, good_boxes), axis=1)  # shape=(300,6)一个类别，一个类别置信度,4个框变量
                    # 添加进result里
                    results[-1].extend(c_pred) #每一类300个框

            if len(results[-1]) > 0:
                # 按照置信度进行排序
                results[-1] = np.array(results[-1])  # shape=(300*num_classes,6)
                #y=np.argsort(x)将x从小到大排序并将其索引输出至y, a[::-1]# 取从后向前的元素
                argsort = np.argsort(results[-1][:, 1])[::-1]  # 取出置信度那一列，并获取从大到小排序的索引
                results[-1] = results[-1][argsort]
                # 选出置信度最大的keep_top_k个
                results[-1] = results[-1][:keep_top_k] # 所有框中选置信度前300的框
        # 获得，在所有预测结果里面，置信度比较高的框
        # 还有，利用先验框和Retinanet的预测结果，处理获得了真实框（预测框）的位置
        return results

    def nms_for_out(self,all_labels,all_confs,all_bboxes,num_classes,nms):
        """
        非极大抑制
        :param all_labels: 建议框的类别
        :param all_confs: 建议框的置信度
        :param all_bboxes: 建议框（left,top,right,bottom)
        :param num_classes: 总类数量
        :param nms: 非极大值抑制的iou阈值
        :return:非极大值抑制完成后剩余的框
        """
        results = []
        if TFVS == 1:
            nms_out = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=nms)
        for c in range(num_classes):
            c_pred = []
            mask = all_labels == c
            if len(all_confs[mask]) > 0:
                # 取出得分高于confidence_threshold的框
                boxes_to_process = all_bboxes[mask]
                confs_to_process = all_confs[mask]
                # 进行iou的非极大抑制
                if TFVS == 1:
                    feed_dict = {self.boxes: boxes_to_process,
                                    self.scores: confs_to_process}
                    idx = self.sess.run(nms_out, feed_dict=feed_dict)
                else:
                    idx = tf.image.non_max_suppression(boxes_to_process, confs_to_process,
                                                           self._top_k,
                                                           iou_threshold=nms)
                # 取出在非极大抑制中效果较好的内容
                good_boxes = boxes_to_process[idx]
                confs = confs_to_process[idx][:, None]
                # 将label、置信度、框的位置进行堆叠。
                labels = c * np.ones((len(idx), 1))
                c_pred = np.concatenate((labels, confs, good_boxes),axis=1)
            results.extend(c_pred)
        return results
