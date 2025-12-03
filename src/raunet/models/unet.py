# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow.keras import backend as K

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm
import tensorflow as tf
from raunet.utils.helper_loss_functions import dice_loss
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from raunet.models.blocks import encoder_block, decoder_block, conv_block


def simple_unet_model(shape=(128, 128, 3),
                                  num_filters=64,
                                  filter_multiplier=[2,3,4,5],
                                  kernel_size=3, dropout=0, batch_norm=True, l_r=0.0001, loss_func=dice_loss):
    # Build the model
    inputs = Input((shape))
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand

    s1, p1 = encoder_block(inputs, num_filters, kernel_size, dropout, batch_norm)
    s2, p2 = encoder_block(p1, num_filters * filter_multiplier[0], kernel_size, dropout, batch_norm)
    s3, p3 = encoder_block(p2, num_filters * filter_multiplier[1], kernel_size, dropout, batch_norm)
    s4, p4 = encoder_block(p3, num_filters * filter_multiplier[2], kernel_size, dropout, batch_norm)

    b1 = conv_block(x=p4, num_filters=num_filters * filter_multiplier[3], kernel_size=3, dropout=0, batch_normalization=True)  # Bridge

    d1 = decoder_block(b1, s4, num_filters * filter_multiplier[2], kernel_size, dropout, batch_norm)
    d2 = decoder_block(d1, s3, num_filters * filter_multiplier[1], kernel_size, dropout, batch_norm)
    d3 = decoder_block(d2, s2, num_filters * filter_multiplier[0], kernel_size, dropout, batch_norm)
    d4 = decoder_block(d3, s1, num_filters, kernel_size, dropout, batch_norm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d4)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_r),
                  loss=[loss_func],
                  metrics=[sm.metrics.IOUScore(threshold=0.5)])

    return model