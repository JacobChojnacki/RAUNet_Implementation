# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import segmentation_models as sm

from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Input, MaxPooling2D, concatenate
from tensorflow.keras.layers import SpatialDropout2D, GlobalMaxPooling2D, UpSampling2D, AveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, normalize
from tensorflow.keras.metrics import IoU, MeanIoU
from tensorflow.keras import backend as K
from Helper_functions.helper_loss_functions import dice_loss


#---------------------------------------------- U-Net ------------------------------------------------------------------#
# def conv_block(x, num_filters, kernel_size, dropout, batch_normalization=False):
#     conv = Conv2D(num_filters, (kernel_size, kernel_size),
#                   padding='same')(x)
#     if batch_normalization:
#         conv = BatchNormalization()(conv)
#     conv = tf.keras.layers.Activation('relu')(conv)
#
#     conv = Conv2D(num_filters, (kernel_size, kernel_size),
#                   padding='same')(conv)
#     if batch_normalization:
#         conv = BatchNormalization()(conv)
#     conv = tf.keras.layers.Activation('relu')(conv)
#
#     if dropout > 0:
#         conv = Dropout(dropout)(conv)
#
#     return conv
#
# def encoder_block(input, num_filters, kernel_size, dropout, batch_normalization):
#     x = conv_block(input, num_filters, kernel_size, dropout, batch_normalization)
#     p = MaxPooling2D((2, 2))(x)
#     return x, p
#
# def decoder_block(input, skip_features, num_filters, kernel_size, dropout, batch_normalization):
#     x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input)
#     x = concatenate([x, skip_features])
#     x = conv_block(x, num_filters, kernel_size, dropout, batch_normalization)
#     return x
#
# def simple_unet_model(shape=(128, 128, 3),
#                                   num_filters=64,
#                                   filter_multiplier=[2,3,4,5],
#                                   kernel_size=3, dropout=0, batch_norm=True, l_r=0.0001, loss_func=dice_loss):
#     # Build the model
#     inputs = Input((shape))
#     # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
#
#     s1, p1 = encoder_block(inputs, num_filters, kernel_size, dropout, batch_norm)
#     s2, p2 = encoder_block(p1, num_filters * filter_multiplier[0], kernel_size, dropout, batch_norm)
#     s3, p3 = encoder_block(p2, num_filters * filter_multiplier[1], kernel_size, dropout, batch_norm)
#     s4, p4 = encoder_block(p3, num_filters * filter_multiplier[2], kernel_size, dropout, batch_norm)
#
#     b1 = conv_block(x=p4, num_filters=num_filters * filter_multiplier[3], kernel_size=3, dropout=0, batch_normalization=True)  # Bridge
#
#     d1 = decoder_block(b1, s4, num_filters * filter_multiplier[2], kernel_size, dropout, batch_norm)
#     d2 = decoder_block(d1, s3, num_filters * filter_multiplier[1], kernel_size, dropout, batch_norm)
#     d3 = decoder_block(d2, s2, num_filters * filter_multiplier[0], kernel_size, dropout, batch_norm)
#     d4 = decoder_block(d3, s1, num_filters, kernel_size, dropout, batch_norm)
#
#     outputs = Conv2D(1, (1, 1), activation='sigmoid')(d4)
#
#     model = Model(inputs=[inputs], outputs=[outputs])
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_r),
#                   loss=[loss_func],
#                   metrics=[sm.metrics.IOUScore(threshold=0.5)])
#
#     return model
#

# ____________________________________________________________________________________________________________________________

# --------------------------------------- RESIDUAL NETWORK PART -----------------------------------------------------#
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
# ------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------------ ATTENTION U-NET--------------------------------------------#
def repeat_elem(tensor, rep):
    """
    Lamda function repeats the elements of a tensor along an axis by a factor of rep
    """
    return tf.keras.layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                                  arguments={'repnum': rep})(tensor)

def gating_signal(input, out_size, batch_normalization=False):
    '''
    Resize the down layer feature map into the samo dimension as the up layer feature map using 1x1 conv
    Returns:
        The gating feature map with the same dimension of the up layer feature map
    '''
    x = Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_normalization:
        x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

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

# def Attention_unet_model(shape=(128, 128, 3),
#                                   num_filters=64,
#                                   filter_multiplier=[2,3,4,5],
#                                   kernel_size=3, dropout=0, batch_norm=True, l_r=0.0001, loss_func=dice_loss):
#     inputs = Input((shape))
#
#     s1, p1 = encoder_block(inputs, num_filters, kernel_size, dropout, batch_norm)
#     s2, p2 = encoder_block(p1, num_filters * filter_multiplier[0], kernel_size, dropout, batch_norm)
#     s3, p3 = encoder_block(p2, num_filters * filter_multiplier[1], kernel_size, dropout, batch_norm)
#     s4, p4 = encoder_block(p3, num_filters * filter_multiplier[2], kernel_size, dropout, batch_norm)
#
#     b1 = conv_block(x=p4, num_filters=num_filters * filter_multiplier[3], kernel_size=3, dropout=0, batch_normalization=True)  # Bridge
#
#     gatting_4 = gating_signal(b1, num_filters * filter_multiplier[2], batch_norm)
#     att_4 = attention_block(s4, gatting_4, num_filters * filter_multiplier[2])
#     d1 = decoder_block(b1, att_4, num_filters * filter_multiplier[2], kernel_size, dropout=0.1, batch_normalization=batch_norm)
#
#     gating_3 = gating_signal(d1, num_filters * filter_multiplier[1], batch_norm)
#     att_3 = attention_block(s3, gating_3, num_filters * filter_multiplier[1])
#     d2 = decoder_block(d1, att_3, num_filters * filter_multiplier[1], kernel_size, dropout=0.1, batch_normalization=batch_norm)
#
#     gating_2 = gating_signal(d2, num_filters * filter_multiplier[0], batch_norm)
#     att_2 = attention_block(s2, gating_2, num_filters * filter_multiplier[0])
#     d3 = decoder_block(d2, att_2, num_filters * filter_multiplier[0], kernel_size, dropout=0.1, batch_normalization=batch_norm)
#
#     gating_1 = gating_signal(d3, num_filters, batch_norm)
#     att_1 = attention_block(s1, gating_1, num_filters)
#     d4 = decoder_block(d3, att_1, num_filters, kernel_size, dropout=0.1, batch_normalization=batch_norm)
#
#     outputs = Conv2D(1, (1, 1), activation='sigmoid')(d4)
#
#     model = Model(inputs=[inputs], outputs=[outputs])
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_r),
#                   loss=[loss_func],
#                   metrics=[sm.metrics.IOUScore(threshold=0.5)])
#
#     return model

#--------------------------------------------------------------------------------------------------------------------#

# ------------------------------------------------ RESIDUAL NETWORK -------------------------------------------------#
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


def Attention_residual_unet_model(shape=(128, 128, 3),
                                  num_filters=64,
                                  filter_multiplier=[2,3,4,5],
                                  kernel_size=3, dropout=0, batch_norm=True, l_r=0.0001, loss_func=dice_loss):
    inputs = Input((shape))

    s1, p1 = encoder_residual_block(inputs, num_filters, kernel_size=kernel_size, batch_norm=batch_norm, dropout=0)
    s2, p2 = encoder_residual_block(p1, filter_multiplier[0] * num_filters, kernel_size=kernel_size, batch_norm=batch_norm, dropout=dropout)
    s3, p3 = encoder_residual_block(p2, filter_multiplier[1] * num_filters, kernel_size=kernel_size, batch_norm=batch_norm, dropout=dropout, dilation_rate=2)
    s4, p4 = encoder_residual_block(p3, filter_multiplier[2] * num_filters, kernel_size=kernel_size, batch_norm=batch_norm, dropout=dropout, dilation_rate=3)

    b1 = res_conv_block(x=p4, num_filters=filter_multiplier[3]*num_filters, kernel_size=kernel_size,
                              dropout=dropout, batch_normalization=batch_norm, dilation_rate=4)  # Bridge

    gatting_4 = gating_signal(b1, filter_multiplier[2] * num_filters, batch_norm)
    att_4 = attention_block(s4, gatting_4, filter_multiplier[2] * num_filters, name="attention_1")
    d1 = decoder_residual_block(b1, att_4, filter_multiplier[2] * num_filters, kernel_size, dropout=dropout, batch_norm=batch_norm)

    gating_3 = gating_signal(d1, filter_multiplier[1] * num_filters, batch_norm)
    att_3 = attention_block(s3, gating_3, filter_multiplier[1] * num_filters, name="attention_2")
    d2 = decoder_residual_block(d1, att_3, filter_multiplier[1] * num_filters, kernel_size, dropout=dropout, batch_norm=batch_norm)

    gating_2 = gating_signal(d2, filter_multiplier[0] * num_filters, batch_norm)
    att_2 = attention_block(s2, gating_2, filter_multiplier[0] * num_filters, name="attention_3")
    d3 = decoder_residual_block(d2, att_2, filter_multiplier[0] * num_filters, kernel_size, dropout=dropout, batch_norm=batch_norm)

    gating_1 = gating_signal(d3, num_filters, batch_norm)
    att_1 = attention_block(s1, gating_1, num_filters, name="attention_4")
    d4 = decoder_residual_block(d3, att_1, num_filters, kernel_size, dropout=0, batch_norm=batch_norm)

    outputs = Conv2D(1, (1, 1))(d4)
    outputs = tf.keras.layers.Activation("sigmoid")(outputs)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_r),
                  loss=[loss_func],
                  metrics=[sm.metrics.IOUScore(threshold=0.5)])
    return model

#----------------------------------------------------------------------------------------------------------------------#
#---------------------------------------- Labels Model Classifier -----------------------------------------------------#
def Classification_model(shape=(128, 128, 1),
                                  num_filters=32,
                                  filter_multiplier=[2,3,4,5],
                                  kernel_size=3,
                                  dropout=0,
                                  batch_norm=True,
                                  l_r=0.001,
                                  loss_func='categorical_crossentropy'):
    inputs = Input((shape))

    # Left
    s1, p1 = encoder_residual_block(inputs, num_filters, kernel_size=kernel_size, batch_norm=batch_norm, dropout=0)

    # Right
    s2, p2 = encoder_residual_block(inputs, num_filters, kernel_size=kernel_size, batch_norm=batch_norm, dropout=0, dilation_rate=2)

    p1 = concatenate([p1, p2])

    s2, p2 = encoder_residual_block(p1, filter_multiplier[0] * num_filters, kernel_size=kernel_size, batch_norm=batch_norm, dropout=dropout)
    s3, p3 = encoder_residual_block(p2, filter_multiplier[1] * num_filters, kernel_size=kernel_size, batch_norm=batch_norm, dropout=dropout)
    s4, p4 = encoder_residual_block(p3, filter_multiplier[2] * num_filters, kernel_size=kernel_size, batch_norm=batch_norm, dropout=dropout)

    b1 = res_conv_block(x=p4, num_filters=filter_multiplier[3]*num_filters, kernel_size=kernel_size,
                              dropout=dropout, batch_normalization=batch_norm)  # Bridge
    x = Flatten()(b1)
    dense = Dense(num_filters)(x)
    dense = BatchNormalization()(x)
    dense = Dropout(0.3)(x)
    dense = tf.keras.layers.Activation('relu')(dense)
    outputs = Dense(3, activation='softmax')(dense)
    model = Model(inputs = [inputs], outputs=[outputs])

    model.compile(loss=loss_func,
                   optimizer=tf.keras.optimizers.Adam(learning_rate=l_r),
                   metrics=['accuracy'])
    return model

# def DenseBlock(x, num_filters, kernel_size, dropout, dilation_rate=(1, 1)):
#     dense_conv = BatchNormalization()(x)
#     dense_conv = tf.keras.layers.Activation("relu")(dense_conv)
#     dense_conv = Conv2D(num_filters, (1,1), padding='same', dilation_rate=dilation_rate)(dense_conv)
#     dense_conv = Dropout(dropout)(dense_conv)
#     dense_conv = BatchNormalization()(dense_conv)
#     dense_conv = tf.keras.layers.Activation('relu')(dense_conv)
#     dense_conv = Dropout(dropout)(dense_conv)
#     dense_conv = Conv2D(num_filters, kernel_size, padding='same', dilation_rate=dilation_rate)(dense_conv)
#     output = concatenate([x, dense_conv])
#     output = tf.keras.layers.Activation('relu')(output)
#     output = Dropout(dropout)(output)
#     return output

