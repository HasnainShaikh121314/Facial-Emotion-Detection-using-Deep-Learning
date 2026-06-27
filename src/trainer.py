"""
Model Training Utilities
"""

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)

from src.config import (
    MODEL_DIR,
    MODEL_PATH,
    OUTPUT_DIR,
    EPOCHS,
    BATCH_SIZE
)


def get_callbacks():
    """
    Create training callbacks.

    Returns
    -------
    list
        List of TensorFlow callbacks.
    """

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    callbacks = [

        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),

        CSVLogger(
            str(OUTPUT_DIR / "training_log.csv"),
            append=False
        )
    ]

    return callbacks


def train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test
):
    """
    Train CNN model.

    Parameters
    ----------
    model : tensorflow.keras.Model

    X_train : ndarray

    y_train : ndarray

    X_test : ndarray

    y_test : ndarray

    Returns
    -------
    tensorflow.keras.callbacks.History
    """

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_test,
            y_test
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=get_callbacks(),

        verbose=1,

        shuffle=True

    )

    return history