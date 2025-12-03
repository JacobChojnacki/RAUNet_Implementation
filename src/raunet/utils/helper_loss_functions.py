from tensorflow.keras import backend as K
from tensorflow.keras import losses
import tensorflow as tf
def jaccard_coef(y_true, y_pred, smooth=1.0):
    """
    Function to calcularde Jaccard Inded definied by equation:
                   | A ∩ B |           | A ∩ B |
        J(A,B) =  ----------- = ---------------------
                  | A ∪ B |     |A| + |B| - | A ∩ B |
    Args:
        y_true (tensor) -> ground true labels
        y_pred (tensor) -> predicitions made by model
    Returns:
        Jaccard_Coef (float32)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def jaccard_loss_function(y_true, y_pred):
    """
    Calculate Jaccard Loss function also known as IoU (loss)
    Args:
        y_true (tensor) -> ground true labels
        y_pred (tensor) -> predicitions made by model
    Returns:
        Jaccard_loss_function (float32), which can be used during backpropagation
    """
    return 1.0-jaccard_coef(y_true, y_pred)


def dice_loss(y_true, y_pred, smooth = 1.):
    """
    Function to calculate dice loss
    Args:
        y_true (tensor) -> ground true labels
        y_pred (tensor) -> predicitions made by model
        smooth (float) -> how much our loss has to be smooth
    Returns:
        Dice_Loss_function (float32)
    """
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)

    intersection = K.sum(y_true * y_pred)
    dice_coef = (2 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    dice_loss = 1.0 - dice_coef
    return dice_loss


