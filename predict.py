"""
Predict facial expression from a single image.

Usage:
    python predict.py --image path/to/image.jpg
"""

import argparse
import cv2
import numpy as np

from tensorflow.keras.models import load_model

from src.config import (
    MODEL_PATH,
    IMAGE_SIZE,
    CLASS_NAMES
)


def preprocess_image(image_path):
    """
    Load and preprocess a single image.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )

    image = cv2.resize(image, IMAGE_SIZE)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    image = image.astype(np.float32) / 255.0

    image = image.reshape(
        1,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        1
    )

    return image


def predict(image_path):

    print("Loading model...")

    model = load_model(MODEL_PATH)

    image = preprocess_image(image_path)

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    class_index = np.argmax(prediction)

    confidence = prediction[class_index]

    print("\nPrediction Result")
    print("-" * 30)

    print(f"Emotion    : {CLASS_NAMES[class_index]}")
    print(f"Confidence : {confidence*100:.2f}%")

    return CLASS_NAMES[class_index], confidence


def main():

    parser = argparse.ArgumentParser(
        description="Facial Expression Recognition"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )

    args = parser.parse_args()

    predict(args.image)


if __name__ == "__main__":
    main()