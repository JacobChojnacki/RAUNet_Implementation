# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
# 1. Set this environment variable BEFORE importing segmentation_models
os.environ["SM_FRAMEWORK"] = "tf.keras"

# 2. Now import segmentation_models
import segmentation_models as sm
import tensorflow as tf
from raunet.utils.helper_loss_functions import dice_loss
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Flatten
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
from raunet.models.blocks import res_conv_block, encoder_residual_block, attention_block, decoder_residual_block
from raunet.models.utils import gating_signal



def attention_residual_unet_model(shape=(128, 128, 3),
                                  num_filters=64,
                                  filter_multiplier=None,
                                  kernel_size=3, dropout=0, batch_norm=True, l_r=0.0001, loss_func=dice_loss):
    if filter_multiplier is None:
        filter_multiplier = [2, 3, 4, 5]
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


#---------------------------------------- Labels Model Classifier -----------------------------------------------------#

def classification_model(shape=(128, 128, 1),
                         num_filters=32,
                         filter_multiplier=None,
                         kernel_size=3,
                         dropout=0,
                         batch_norm=True,
                         l_r=0.001,
                         loss_func='categorical_crossentropy'):
    if filter_multiplier is None:
        filter_multiplier = [2, 3, 4, 5]
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
