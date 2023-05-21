import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tf_explain.core.grad_cam import GradCAM


def loadImage(image_path, shape):
    image = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(image_path))
    image = cv2.resize(image, (shape, shape))
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image


def loadImageV2(image_path, size):
    """
    Specify the wat in which we load the data from PATH.
    Args:
        image_path (string) -> The location of our data
        size (int) -> It defines to size loaded data (IMAGE_HEIGHT, IMAGE_WIDTH), IMAGE_HEIGHT=IMAGE_WIDTH
    Returns:
        Image (array) - Loaded image with determined by argument size
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     image = cv2.resize(image, (shape, shape))
    image = Image.fromarray(image)
    image = image.resize((size, size))
    return image


def num(filename):
    """
    Retrieves a number from a specified path string ^.*?\(\d+).*$
    Args:
        filename (string) -> path, which we want to check
    Returns:
        val (int) -> number between parentheses
    """
    val = 0
    for i in range(len(filename)):
        if filename[i] == '(':
            while True:
                i += 1
                if filename[i] == ')':
                    break
                val = (val * 10) + int(filename[i])
            break
    return val


def createDataset(images, masks, batch_size):
    """
    Loads images and masks and divide them into batches to create dataset, which can be used in training process.
    It is imporant to load equal number of images and masks.
    Args:
        Images (array) -> CORRESPONDING image to mask
        Masks (array) -> CORRESPONDING mask to image
    Returns:
        Dataset (batch)
    """
    data = tf.data.Dataset.from_tensor_slices((images, masks))
    data = data.shuffle(1000).batch(batch_size, drop_remainder=True)
    data = data.prefetch(tf.data.AUTOTUNE)
    return data


def loadData(path=None, size=256, files=False, batch_size=16, split=0.15, classification_problem=False):
    '''
    Function to load data from specific folders architecure and in defined shape.
    Folders Architecture:
        benign: masks + images
        malignant: masks + images
        normals: masks + images
    If there is more masks to one image, it joins them together. It additionaly returns informaion about
    size in MGB.
    Args:
        path (string) -> The location of our data.
        size (int) -> It defines to size loaded data (IMAGE_HEIGHT, IMAGE_WIDTH), IMAGE_HEIGHT=IMAGE_WIDTH.
        batch_size (int) -> The number of images that make up one data partition.
        split (float32) -> The size of the test set (0-1).
        files (boolean) -> Defines whether to return the same datataset(train and test) or with data (train, test)
    Returns:
        train_data (batch) -> dataset to training process
        test_data (batch) -> dataset to testing process
        X_train (array) -> Split training images
        X_test (array) -> Split test images
        y_train (array) -> Split train masks
        y_test (array) -> Split test masks
    '''

    classes = os.listdir(path)
    benign_path = []
    malignant_path = []
    normal_path = []
    images_paths = []
    labels = []

    for single_class in classes:
        images_and_mask = glob(os.path.join(path, "", single_class, "*"))
        if single_class == "benign":
            benign_path += images_and_mask
        elif single_class == "malignant":
            malignant_path += images_and_mask
        else:
            normal_path += images_and_mask

    benign_image, benign_mask = np.zeros((437, size, size, 1)), \
        np.zeros((437, size, size, 1))

    malignant_image, malignant_mask = np.zeros((210, size, size, 1)), \
        np.zeros((210, size, size, 1))

    normal_image, normal_mask = np.zeros((133, size, size, 1)), \
        np.zeros((133, size, size, 1))

    for image in benign_path:
        img = loadImageV2(image, size)
        if "mask" in image:
            benign_mask[num(image) - 1] += img_to_array(img)
        else:
            benign_image[num(image) - 1] += img_to_array(img)
            labels.append(0)
            images_paths.append(image)

    for image in malignant_path:
        img = loadImageV2(image, size)
        if "mask" in image:
            malignant_mask[num(image) - 1] += img_to_array(img)
        else:
            malignant_image[num(image) - 1] += img_to_array(img)
            labels.append(1)
            images_paths.append(image)

    for image in normal_path:
        img = loadImageV2(image, size)
        if "mask" in image:
            normal_mask[num(image) - 1] += img_to_array(img)
        else:
            normal_image[num(image) - 1] += img_to_array(img)
            labels.append(2)
            images_paths.append(image)

    images = np.concatenate((benign_image, malignant_image, normal_image), axis=0).astype(np.float32) / 255.

    masks = np.concatenate((benign_mask, malignant_mask, normal_mask), axis=0).astype(np.float32) / 255.
    masks[masks > 1.0] = 1.0

    if classification_problem:
        return images, masks, labels, images_paths

    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=split)

    train_data = createDataset(X_train, y_train, batch_size)
    test_data = createDataset(X_test, y_test, batch_size)

    print("Used memory to store the float image dataset is: ", images.nbytes / (1024 * 1024))
    print("Used memory to store the float mask dataset is: ", masks.nbytes / (1024 * 1024))

    if files:
        return train_data, test_data, X_train, X_test, y_train, y_test

    return train_data, test_data


def show_images(data, model=None, explain=False, n_images=5, SIZE=(30, 15)):
    """
    Plot images with corresponding masks from dataset. It also puts them together to visualize.
    We can also load our own custom model to see comparision between predicted mask and ground truth. It shows
    us where model pay the most attention to make prediction.
    Args:
        data (batch) - Dataset with images and masks
        model (tensor) - Model trainded by user
        explain (boolean) - It determinates if we want to visualize model predictions
        n_images (int) - Amount images to visualize
        SIZE (tuple):int -> Size of figure
    Returns:
        Different kinds of visualizations
    """
    # Plot Configurations
    if model is not None:
        if explain:
            n_cols = 5
        else:
            n_cols = 4
    else:
        n_cols = 3

    # Iterate through data
    for plot_no in range(1, n_images + 1):
        for index, (images, masks) in enumerate(iter(data)):

            # Select Items
            id = np.random.randint(len(images))
            image, mask = images[id], masks[id]

            if model is not None:

                if explain:
                    # Make Prediction
                    pred_mask = model.predict(image[np.newaxis, ...])[0]

                    # Grad CAM
                    cam = GradCAM()
                    cam = cam.explain(
                        validation_data=(np.array(image[np.newaxis, ...]), np.array(mask)),
                        class_index=1,
                        layer_name='attention_4',
                        model=model
                    )

                    # Figure
                    plt.figure(figsize=SIZE)

                    # Original Image
                    plt.subplot(1, n_cols, 1)
                    plt.imshow(image, cmap='gray')
                    plt.axis('off')
                    plt.title("Original Image")

                    # Original Mask
                    plt.subplot(1, n_cols, 2)
                    plt.imshow(mask, cmap='gray')
                    plt.axis('off')
                    plt.title('Original Mask')

                    # Predicted Mask
                    plt.subplot(1, n_cols, 3)
                    plt.imshow(pred_mask, cmap='gray')
                    plt.axis('off')
                    plt.title("Predicted Mask")

                    # Mixed
                    plt.subplot(1, n_cols, 4)
                    plt.imshow(image, cmap='gray')
                    plt.imshow(pred_mask, alpha=0.4)
                    plt.axis('off')
                    plt.title("Overlaped Prediction")

                    # Grad Cam
                    plt.subplot(1, n_cols, 5)
                    plt.imshow(cam)
                    plt.axis('off')
                    plt.title("Grad CAM")

                    # Show Image
                    plt.show()

            else:

                # Figure
                plt.figure(figsize=SIZE)

                # Original Image
                plt.subplot(1, n_cols, 1)
                plt.imshow(image, cmap='gray')
                plt.axis('off')
                plt.title("Original Image")

                # Original Mask
                plt.subplot(1, n_cols, 2)
                plt.imshow(mask, cmap='gray')
                plt.axis('off')
                plt.title('Original Mask')

                # Mixed
                plt.subplot(1, n_cols, 3)
                plt.imshow(image, cmap='gray')
                plt.imshow(mask, alpha=0.4, cmap='gray')
                plt.axis('off')
                plt.title("Overlaped Image")

                # Show Image
                plt.show()

            # Break Loop
            break


def preprocess_classification_data(masks, labels):
    labels = np.array(labels)
    labels = to_categorical(labels)
    X_train, X_test, y_train, y_test = train_test_split(masks,
                                                        labels,
                                                        test_size=0.10,
                                                        shuffle=True,
                                                        random_state=1)
    train_gen = ImageDataGenerator(horizontal_flip=True,
                                   rotation_range=15,
                                   width_shift_range=[-10, 10],
                                   height_shift_range=[-10, 10],
                                   zoom_range=[0.80, 1.00])
    train_gen.fit(X_train)
    return train_gen, X_train, X_test, y_train, y_test


def visualize_segmentation_and_classification(images, masks, labels, model_c, model_s, n_images=10):
    # labels = np.array(labels)
    # labels = to_categorical(labels)

    y_masks = model_s.predict(images) > 0.5
    y_labels = model_c.predict(images) > 0.8
    info = ['benign', 'malignant', 'normal']

    plt.figure(figsize=(40, 80))
    i = 0
    amount = 0
    while amount < n_images:
        x = np.random.randint(0, len(images) - 1)
        # Grad CAM
        cam = GradCAM()
        cam = cam.explain(
            validation_data=(np.array(images[x][np.newaxis, ...]), np.array(masks[x])),
            class_index=1,
            layer_name='attention_4',
            model=model_s
        )

        plt.subplot(n_images, 4, i + 1)
        plt.imshow(images[x], 'gray')
        plt.title(f'{info[np.argmax(labels[x])]}', fontsize=45)
        plt.axis('off')

        plt.subplot(n_images, 4, i + 2)
        plt.imshow(masks[x], 'gray')
        plt.title(f'{info[np.argmax(labels[x])]}', fontsize=45)
        plt.axis('off')

        plt.subplot(n_images, 4, i + 3)
        plt.imshow(y_masks[x], 'gray')
        plt.title(f'{info[np.argmax(y_labels[x])]}', fontsize=45)
        plt.axis('off')

        # Grad Cam
        plt.subplot(n_images, 4, i + 4)
        plt.imshow(cam)
        plt.axis('off')
        plt.title("Grad CAM", fontsize=45)

        amount += 1
        i += 4


plt.show()