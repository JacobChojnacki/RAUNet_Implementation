
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
# 1. Set this environment variable BEFORE importing segmentation_models
os.environ["SM_FRAMEWORK"] = "tf.keras"

# 2. Now import segmentation_models
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization


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