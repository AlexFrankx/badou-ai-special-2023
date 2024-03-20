# from keras.engine.topology import Layer 高版本keras不支持注释掉，改为下一行即可。
from keras.layers import Layer
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf


class RoiPoolingConv(Layer):
    '''
    ROI pooling layer for 2D inputs.
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''

    def __init__(self, pool_size, num_rois, **kwargs):
        """
        新版keras
        K.image_data_format() == 'channels_last'
        K.image_data_format() == 'channels_first'
        替代
        K.image_dim_ordering() == 'tf'
        K.image_dim_ordering() == 'th'
        """

        # self.dim_ordering = K.image_dim_ordering()
        # assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.dim_ordering = K.image_data_format()  # 默认值为channels_last
        assert self.dim_ordering in {'channels_last', 'channels_first'}, 'dim_ordering must be in {channels_last, channels_first}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert(len(x) == 2)
        # 进入RoiPooling层的数据包括cnn之后的featuremaps+经过RPN后的建议框
        img = x[0]
        rois = x[1]

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # 将建议框缩放为pool_size * pool_size
            # rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))  # 高版本不支持，改为下一行即可
            rs = tf.image.resize(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))

            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4)) # 用法与np.transpose()一致，转置，当前维度正确，可以不加

        return final_output
