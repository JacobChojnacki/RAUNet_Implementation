import matplotlib.pyplot as plt
import numpy as np


def visualize_predictions(x, y_true, y_pred, num_samples=10, figsize=(15, 4)):
    """
    Visualizes Input, Ground Truth, and Prediction side-by-side.

    Parameters:
    - X: Array of input images
    - y_true: Array of ground truth images
    - y_pred: Array of predicted images
    - num_samples: How many rows of images to show (default: 10)
    - figsize: Tuple representing the size of ONE row (default: 15 wide, 4 high)
    """

    # Calculate total figure height based on number of samples
    total_height = figsize[1] * num_samples
    plt.figure(figsize=(figsize[0], total_height))

    for i in range(num_samples):
        # 1. Input Image
        # Logic: num_samples rows, 3 columns, position index
        plt.subplot(num_samples, 3, (i * 3) + 1)
        plt.imshow(x[i], cmap='gray')
        plt.title('Real Medic Image')
        plt.axis('off')

        # 2. Ground Truth
        plt.subplot(num_samples, 3, (i * 3) + 2)
        plt.imshow(y_true[i], cmap='gray')
        plt.title('Ground Truth Img')
        plt.axis('off')

        # 3. Prediction
        plt.subplot(num_samples, 3, (i * 3) + 3)
        plt.imshow(y_pred[i], cmap='gray')
        plt.title('Predicted Image')
        plt.axis('off')

    plt.tight_layout()
    plt.show()



def visualize_random_samples(x, y, class_labels, num_rows=5, num_cols=5, figsize=(20, 20)):
    """
    Plots a grid of random images from the dataset with their corresponding labels.

    Parameters:
    - x: The array of image data (e.g., X_train_c)
    - y: The array of labels (e.g., y_train_c, one-hot encoded)
    - class_labels: List or dictionary mapping indices to class names (e.g., 'info')
    - num_rows: Number of rows in the grid (default: 5)
    - num_cols: Number of columns in the grid (default: 5)
    - figsize: Tuple for figure dimensions (default: 20x20)
    """

    dataset_size = len(x)
    total_plots = num_rows * num_cols

    plt.figure(figsize=figsize)

    for i in range(total_plots):
        j = np.random.randint(0, dataset_size)

        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(x[j], cmap='gray')

        label_index = np.argmax(y[j])
        label_name = class_labels[label_index]

        plt.title(f'{label_name}', fontsize=15)
        plt.axis('off')

    plt.tight_layout()
    plt.show()