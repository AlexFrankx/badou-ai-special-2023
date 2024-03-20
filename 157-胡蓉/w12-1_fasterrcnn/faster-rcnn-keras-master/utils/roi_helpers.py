import numpy as np
import pdb
import math
import copy
import time

def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h

def iou(a, b):
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)

def calc_iou(R, config, all_boxes, width, height, num_classes):
    # print(all_boxes)
    bboxes = all_boxes[:,:4]   # 所有的目标框
    gta = np.zeros((len(bboxes), 4))
    for bbox_num, bbox in enumerate(bboxes):
        # featuremap的大小是原图的1/config.rpn_stride倍，从归一画坐标转换到原图坐标再转换到featuremaps坐标
        gta[bbox_num, 0] = int(round(bbox[0]*width/config.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox[1]*height/config.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox[2]*width/config.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox[3]*height/config.rpn_stride))
    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = []
    # print(gta)
    for ix in range(R.shape[0]):
        # 将每一个建议框都转换到到featuremap坐标
        x1 = R[ix, 0]*width/config.rpn_stride
        y1 = R[ix, 1]*height/config.rpn_stride
        x2 = R[ix, 2]*width/config.rpn_stride
        y2 = R[ix, 3]*height/config.rpn_stride
        # 四舍五入取整
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        # print([x1, y1, x2, y2])
        best_iou = 0.0
        best_bbox = -1
        for bbox_num in range(len(bboxes)):
            # 每一个建议框分别与每个目标框计算Iou
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 1], gta[bbox_num, 2], gta[bbox_num, 3]], [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num
        # print(best_iou)
        if best_iou < config.classifier_min_overlap:
            # 当best_iou在小于0.1，则认为未框到样本，舍弃该框
            continue
        else:
            w = x2 - x1  # 建议框的宽
            h = y2 - y1 # 建议框的高
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if config.classifier_min_overlap <= best_iou < config.classifier_max_overlap:
                # 当best_iou在0.1~0.5之间，label=-1
                label = -1
            elif config.classifier_max_overlap <= best_iou:
                # 当best_iou在大于0.5，label取best_iou时的框的label，即最匹配的框
                label = int(all_boxes[best_bbox, -1])
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 2]) / 2.0  # 最匹配的框的中心点坐标
                cyg = (gta[best_bbox, 1] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0  # 最匹配的框的左上点坐标
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)  # 建议框的x修正系数
                ty = (cyg - cy) / float(h)  # 建议框的y修正系数
                tw = np.log((gta[best_bbox, 2] - gta[best_bbox, 0]) / float(w))  # 建议框的w修正系数
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 1]) / float(h))  # 建议框的h修正系数
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError
        # 此刻的label只有1和-1
        # print(label)
        class_label = num_classes * [0]  # 建立一个与num_classes（所有类别+置信度）相同数量的列表，并全部赋值为0
        class_label[label] = 1  # 仅为判断为该类别的位置赋1
        y_class_num.append(copy.deepcopy(class_label))  # 将所有框的类别结果放入y_class_num中
        coords = [0] * 4 * (num_classes - 1)  # 所有框的坐标变化量
        labels = [0] * 4 * (num_classes - 1)  # 所有框的分类置信度
        if label != -1:  # 当前建议框有目标，即label=1
            label_pos = 4 * label
            sx, sy, sw, sh = config.classifier_regr_std
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:# 即label=-1
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0: #未检测到目标
        return None, None, None, None

    X = np.array(x_roi)
    # print(X)
    Y1 = np.array(y_class_num)
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs