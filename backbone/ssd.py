"""
构建骨干网络
"""


import numpy as np

from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, ELU, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


def identity_layer(tensor):
    return tensor


def input_mean_normalization(tensor, subtract_mean):
    return tensor - np.array(subtract_mean)


def input_stddev_normalization(tensor,divide_by_stddev):
    return tensor / np.array(divide_by_stddev)


def input_channel_swap(tensor,swap_channels):
    if len(swap_channels) == 3:
        return K.stack([tensor[..., swap_channels[0]], tensor[...,
                                                              swap_channels[1]], tensor[..., swap_channels[2]]],
                       axis=-1)
    elif len(swap_channels) == 4:
        return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]],
                        tensor[..., swap_channels[2]], tensor[..., swap_channels[3]]], axis=-1)


############################################################################


def build_base_ssd_7(image_size,
                normalize_coords=False,
                l2_regularization=0.0,
                subtract_mean=None,
                divide_by_stddev=None,
                swap_channels=False,


                ):
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    l2_reg = l2_regularization
    x = Input(shape=(img_height, img_width, img_channels))
    # The following identity layer is only needed so that the subsequent
    # lambda layers can be optional.
    x1 = Lambda(identity_layer,output_shape=(img_height,img_width,img_channels),name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization,output_shape=(img_height,img_width,img_channels),name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization,output_shape=(img_height,img_width,img_channels),name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap,output_shape=(img_height,img_width,img_channels),name='input_channel_swap')(x1)

    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(
            x1)

    conv1 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv1')(x1)
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(
        conv1)  # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
    conv1 = ELU(name='elu1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    conv2 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv2')(pool1)
    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    conv2 = ELU(name='elu2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    conv3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv3')(pool2)
    conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    conv3 = ELU(name='elu3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv4')(pool3)
    conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    conv4 = ELU(name='elu4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

    conv5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv5')(pool4)
    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    conv5 = ELU(name='elu5')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

    conv6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv6')(pool5)
    conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    conv6 = ELU(name='elu6')(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)

    conv7 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv7')(pool6)
    conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    conv7 = ELU(name='elu7')(conv7)

    base_model = Model(inputs=x, outputs=conv7)
    return base_model

if __name__=="__main__":
    model = build_ssd_7((300,300,3))
    model.summary()