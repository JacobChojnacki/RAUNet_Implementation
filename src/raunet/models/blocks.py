# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
# 1. Set this environment variable BEFORE importing segmentation_models
os.environ["SM_FRAMEWORK"] = "tf.keras"

# 2. Now import segmentation_models
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, concatenate
from raunet.models.utils import repeat_elem

def conv_block(x, num_filters, kernel_size, dropout, batch_normalization=False):
    conv = Conv2D(num_filters, (kernel_size, kernel_size),
                  padding='same')(x)
    if batch_normalization:
        conv = BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)

    conv = Conv2D(num_filters, (kernel_size, kernel_size),
                  padding='same')(conv)
    if batch_normalization:
        conv = BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)

    if dropout > 0:
        conv = Dropout(dropout)(conv)

    return conv

def encoder_block(input, num_filters, kernel_size, dropout, batch_normalization):
    x = conv_block(input, num_filters, kernel_size, dropout, batch_normalization)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters, kernel_size, dropout, batch_normalization):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input)
    x = concatenate([x, skip_features])
    x = conv_block(x, num_filters, kernel_size, dropout, batch_normalization)
    return x


def res_conv_block(x, num_filters, kernel_size, dropout, batch_normalization=True, dilation_rate=(1, 1)):
    conv = Conv2D(num_filters, (kernel_size, kernel_size),
                  padding='same', dilation_rate=dilation_rate)(x)
    if batch_normalization:
        conv = BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)

    if dropout > 0:
        conv = Dropout(dropout)(conv)

    conv = Conv2D(num_filters, (kernel_size, kernel_size),
                  padding='same', dilation_rate=dilation_rate)(conv)
    if batch_normalization:
        conv = BatchNormalization()(conv)


    shorcut = Conv2D(num_filters, kernel_size=(1, 1), padding='same')(x)
    if batch_normalization:
        shortcut = BatchNormalization()(shorcut)
    res_path = tf.keras.layers.add([shorcut, conv])
    res_path = tf.keras.layers.Activation('relu')(res_path)
    return res_path

def attention_block(x, gating, inter_shape, name="layer"):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = tf.keras.layers.add([upsample_g, theta_x])
    act_xg = tf.keras.layers.Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = tf.keras.layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = tf.keras.layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = tf.keras.layers.multiply([upsample_psi, x])

    result = Conv2D(shape_x[3], (1, 1), padding='same', name=name)(y)
    result_bn = BatchNormalization()(result)
    return result_bn


def encoder_residual_block(input, num_filters, kernel_size=3,batch_norm=True, dropout=0, dilation_rate=(1,1)):
    x = res_conv_block(x=input,
                       num_filters=num_filters,
                       kernel_size=kernel_size,
                       dropout=dropout,
                       batch_normalization=batch_norm,
                       dilation_rate=dilation_rate)

    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_residual_block(input, skip_features, num_filters, kernel_size=3, batch_norm=True, dropout=0):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input)
    x = concatenate([x, skip_features])
    x = res_conv_block(x=x, num_filters=num_filters, kernel_size=kernel_size,
                            dropout=dropout,
                            batch_normalization=batch_norm)
    return x

