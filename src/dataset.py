"""
Dataset loading utilities.
"""

import os
import cv2
import numpy as np

from src.config import IMAGE_SIZE

# ==========================================================
# Folder Name → Label Mapping
# ==========================================================

EMOTION_TO_LABEL = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}


def load_dataset(dataset_path):
    """
    Load images from a folder dataset.

    Parameters
    ----------
    dataset_path : str or Path
        Path to train/ or test/ folder.

    Returns
    -------
    images : numpy.ndarray
    labels : numpy.ndarray
    """

    images = []
    labels = []

    for emotion in sorted(os.listdir(dataset_path)):

        emotion_folder = os.path.join(dataset_path, emotion)

        if not os.path.isdir(emotion_folder):
            continue

        if emotion.lower() not in EMOTION_TO_LABEL:
            print(f"Skipping unknown folder: {emotion}")
            continue

        label = EMOTION_TO_LABEL[emotion.lower()]

        print(f"Loading {emotion}...")

        for image_name in os.listdir(emotion_folder):

            image_path = os.path.join(
                emotion_folder,
                image_name
            )

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = cv2.resize(
                image,
                IMAGE_SIZE
            )

            images.append(image)

            labels.append(label)

    images = np.array(images)

    labels = np.array(labels)

    print(f"\nLoaded {len(images)} images.")

    return images, labels