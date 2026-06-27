"""
Image preprocessing utilities.
"""

import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

from src.config import (
    IMAGE_SIZE,
    CHANNELS,
    NUM_CLASSES
)


def rgb_to_grayscale(images):
    """
    Convert RGB/BGR images to grayscale.

    Parameters
    ----------
    images : numpy.ndarray
        Shape (N, H, W, 3)

    Returns
    -------
    numpy.ndarray
        Shape (N, H, W)
    """

    gray_images = []

    for image in images:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray_images.append(gray)

    return np.array(gray_images)


def normalize(images):
    """
    Normalize pixel values to [0,1].
    """

    return images.astype(np.float32) / 255.0


def reshape_images(images):
    """
    Reshape images for CNN input.

    Output:
    (N, 48, 48, 1)
    """

    return images.reshape(

        images.shape[0],

        IMAGE_SIZE[0],

        IMAGE_SIZE[1],

        CHANNELS

    )


def encode_labels(labels):
    """
    One-hot encode labels.
    """

    return to_categorical(
        labels,
        NUM_CLASSES
    )


def preprocess_dataset(images, labels):
    """
    Complete preprocessing pipeline.
    """

   
    images = normalize(images)

    images = reshape_images(images)

    labels = encode_labels(labels)

    return images, labels