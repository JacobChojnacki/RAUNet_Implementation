# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
# 1. Set this environment variable BEFORE importing segmentation_models
os.environ["SM_FRAMEWORK"] = "tf.keras"

# 2. Now import segmentation_models
import segmentation_models as sm
import tensorflow as tf
from raunet.utils.helper_loss_functions import dice_loss
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, BatchNormalization, Flatten
from tensorflow.keras.layers import Input, MaxPooling2D, concatenate
from tensorflow.keras.models import Model
from raunet.models.blocks import attention_block, encoder_block, decoder_block, conv_block
from raunet.models.utils import gating_signal


def Attention_unet_model(shape=(128, 128, 3),
                                  num_filters=64,
                                  filter_multiplier=[2,3,4,5],
                                  kernel_size=3, dropout=0, batch_norm=True, l_r=0.0001, loss_func=dice_loss):
    inputs = Input((shape))

    s1, p1 = encoder_block(inputs, num_filters, kernel_size, dropout, batch_norm)
    s2, p2 = encoder_block(p1, num_filters * filter_multiplier[0], kernel_size, dropout, batch_norm)
    s3, p3 = encoder_block(p2, num_filters * filter_multiplier[1], kernel_size, dropout, batch_norm)
    s4, p4 = encoder_block(p3, num_filters * filter_multiplier[2], kernel_size, dropout, batch_norm)

    b1 = conv_block(x=p4, num_filters=num_filters * filter_multiplier[3], kernel_size=3, dropout=0, batch_normalization=True)  # Bridge

    gatting_4 = gating_signal(b1, num_filters * filter_multiplier[2], batch_norm)
    att_4 = attention_block(s4, gatting_4, num_filters * filter_multiplier[2])
    d1 = decoder_block(b1, att_4, num_filters * filter_multiplier[2], kernel_size, dropout=0.1, batch_normalization=batch_norm)

    gating_3 = gating_signal(d1, num_filters * filter_multiplier[1], batch_norm)
    att_3 = attention_block(s3, gating_3, num_filters * filter_multiplier[1])
    d2 = decoder_block(d1, att_3, num_filters * filter_multiplier[1], kernel_size, dropout=0.1, batch_normalization=batch_norm)

    gating_2 = gating_signal(d2, num_filters * filter_multiplier[0], batch_norm)
    att_2 = attention_block(s2, gating_2, num_filters * filter_multiplier[0])
    d3 = decoder_block(d2, att_2, num_filters * filter_multiplier[0], kernel_size, dropout=0.1, batch_normalization=batch_norm)

    gating_1 = gating_signal(d3, num_filters, batch_norm)
    att_1 = attention_block(s1, gating_1, num_filters)
    d4 = decoder_block(d3, att_1, num_filters, kernel_size, dropout=0.1, batch_normalization=batch_norm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d4)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_r),
                  loss=[loss_func],
                  metrics=[sm.metrics.IOUScore(threshold=0.5)])

    return model

