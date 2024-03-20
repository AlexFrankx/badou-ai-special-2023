import sys

sys.path.append("..")  # 添加上级目录进环境变量，适配当前文件作为main文件运行
from nets.resnet import ResNet50, classifier_layers
from nets.RoiPoolingConv import RoiPoolingConv
from utils.config import Config
from keras.layers import Conv2D, Input, TimeDistributed, Flatten, Dense, Reshape
from keras.models import Model


# 创建建议框网络
# 该网络结果会对先验框进行调整获得建议框
def get_rpn(base_layers, num_anchors):
    """
    创建rpn网络
    base_layers：resnet50输出的特征层（None,38,38,1024）
    num_anchors：先验框框数量，通常为9，即每个网格分配有9个先验框
   """

    # 利用一个512通道的3x3卷积进行特征整合 shape=(None,38,38,512)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)

    # 利用一个1x1卷积调整通道数，获得预测结果
    # rpn_class只预测该先验框是否包含物体 (None,38,38,9)
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    # 预测每个先验框的变化量，4代表变化量的x,y,w,h (None,38,38,36)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    # (None,12996,1) 38x38x9=12996
    x_class = Reshape((-1, 1), name="classification")(x_class)
    # (None,12996,4) 38x38x9=12996
    x_regr = Reshape((-1, 4), name="regression")(x_regr)
    return [x_class, x_regr, base_layers]


def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])  # 该层的输入为feature maps和roi的坐标信息
    out = classifier_layers(out_roi_pool, input_shape=input_shape,
                            trainable=True)  # 输出的是（None, num_riois, 2048)的feature map
    out = TimeDistributed(Flatten())(out)  # 因为是对num_rois个feature maps分别处理的，所以需要使用timedistributed进行包装
    # 我们可以使用包装器TimeDistributed包装Dense，以产生针对各个时间步信号的独立全连接
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]  # 一共有num_rois个out_class和out_regr


def get_frcnn_train_model(config, num_classes):
    inputs = Input(shape=(None, None, 3))  # 支持不同分辨率输入
    roi_input = Input(shape=(None, 4))
    # inputs = Input(shape=(600, 600, 3))
    # roi_input = Input(shape=(32, 4))
    base_layers = ResNet50(inputs)

    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn[:2])

    classifier = get_classifier(base_layers, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier = Model([inputs, roi_input], classifier)

    model_all = Model([inputs, roi_input], rpn[:2] + classifier)
    return model_rpn, model_classifier, model_all


def get_frcnn_predict_model(config, num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None, None, 1024))

    base_layers = ResNet50(inputs)
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn)

    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    return model_rpn, model_classifier_only


def print_summary2file(model, filename):
    """
    将网络结构信息打印到单独文件中
    """

    def log_model_summary(txt):
        with open(filename, 'a+') as f:
            print(txt, file=f)
        f.close()

    model.summary(line_length=120, positions=[.35, .66, .78, 1.], print_fn=log_model_summary)


if __name__ == "__main__":
    config = Config()  # 一些通用参数配置
    NUM_CLASSES = 21  # 分类数量+1(置信度）
    model_rpn, model_classifier, model_all = get_frcnn_train_model(config, NUM_CLASSES)

    # line_length :打印模型的字符总宽度，positions：四列分割线所在的宽度百分比位置，增加宽度可让未显示出来的字符显现
    model_rpn.summary(line_length=120, positions=[.35, .66, .78, 1.])
    model_classifier.summary(line_length=120, positions=[.35, .66, .78, 1.])
    model_all.summary(line_length=120, positions=[.35, .66, .78, 1.])

    # 模型结构打印到文件中，方便单个网络结构查看
    print_summary2file(model_rpn, 'model_rpn_summary.txt')
    print_summary2file(model_classifier, 'model_classifier_summary.txt')
    print_summary2file(model_all, 'model_all_summary.txt')
