"""
CNN Model Definition for Facial Expression Recognition
"""


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
)

from src.config import (
    IMAGE_SIZE,
    CHANNELS,
    NUM_CLASSES,
    LEARNING_RATE
)


def build_model():
    """
    Build and compile the CNN model.

    Returns
    -------
    tensorflow.keras.Model
        Compiled CNN model.
    """

    
    model = Sequential(name="FaceExpressionCNN")
    data_augmentation = tf.keras.Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(0.05),
            RandomZoom(0.05),
        ],
        name="data_augmentation",
    )
    model.add(
        Input(
            shape=(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                CHANNELS
            )
        )
    )
    model.add(data_augmentation)
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        )
    )
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        )
    )

    model.add(BatchNormalization())

    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    model.add(Dropout(0.25))

    # ==========================================================
    # Convolution Block 2
    # ==========================================================
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        )
    )

    model.add(BatchNormalization())

    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        )
    )

    model.add(BatchNormalization())

    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    model.add(Dropout(0.25))

    # ==========================================================
    # Convolution Block 3
    # ==========================================================
    model.add(
        Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        )
    )

    model.add(BatchNormalization())

    model.add(
        Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        )
    )

    model.add(BatchNormalization())

    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    model.add(Dropout(0.25))

    # ==========================================================
    # Fully Connected Layers
    # ==========================================================
    model.add(Flatten())

    model.add(
        Dense(
            512,
            activation="relu"
        )
    )

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    # ==========================================================
    # Output Layer
    # ==========================================================
    model.add(
        Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    )

    # ==========================================================
    # Compile Model
    # ==========================================================
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()