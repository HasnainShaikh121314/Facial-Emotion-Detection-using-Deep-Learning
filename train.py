"""
Main Training Script

Usage:
    python train.py
"""

from src.config import TRAIN_DIR, TEST_DIR
from src.dataset import load_dataset
from src.preprocessing import preprocess_dataset
from src.model import build_model
from src.trainer import train_model
from src.visualization import plot_training_history


def main():

    print("=" * 60)
    print(" Facial Expression Recognition Training ")
    print("=" * 60)

    # ----------------------------------------------------
    # Load Dataset
    # ----------------------------------------------------

    print("\nLoading training dataset...")

    X_train, y_train = load_dataset(TRAIN_DIR)

    print("Training samples :", len(X_train))

    print("\nLoading testing dataset...")

    X_test, y_test = load_dataset(TEST_DIR)

    print("Testing samples :", len(X_test))

    # ----------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------

    print("\nPreprocessing datasets...")

    X_train, y_train = preprocess_dataset(
        X_train,
        y_train
    )

    X_test, y_test = preprocess_dataset(
        X_test,
        y_test
    )

    print("Training shape :", X_train.shape)
    print("Testing shape  :", X_test.shape)

    # ----------------------------------------------------
    # Build Model
    # ----------------------------------------------------

    print("\nBuilding CNN model...")

    model = build_model()

    model.summary()

    # ----------------------------------------------------
    # Train
    # ----------------------------------------------------

    print("\nTraining started...\n")

    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    print("\nTraining completed!")

    # ----------------------------------------------------
    # Save Training Curves
    # ----------------------------------------------------

    plot_training_history(history)

    print("\nTraining plots saved.")

    print("\nBest model saved successfully.")


if __name__ == "__main__":
    main()