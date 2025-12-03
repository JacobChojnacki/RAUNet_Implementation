from raunet.utils.helper_loss_functions import jaccard_loss_function

hyperparameters = {
    "batch_size" : 32,
    "num_filters": 16,
    "filter_multiplier" : [4,4,8,16],
    "dropout": 0.2,
    "batch_norm": True,
    "kernel_size": 3,
    "epochs": 500,
    "learning_rate": 0.001,
    "loss_func": jaccard_loss_function,
    "size": 128,
    "shape": (128, 128, 1)
}