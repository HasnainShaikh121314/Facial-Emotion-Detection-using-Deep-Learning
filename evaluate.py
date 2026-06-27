"""
Evaluate the trained model.

Usage:
    python evaluate.py
"""

from tensorflow.keras.models import load_model

from src.config import (
    TEST_DIR,
    MODEL_PATH
)

from src.dataset import load_dataset
from src.preprocessing import preprocess_dataset
from src.metrics import evaluate_model


def main():

    print("=" * 60)
    print(" Facial Expression Recognition Evaluation ")
    print("=" * 60)

    # ----------------------------------------------------
    # Load Test Dataset
    # ----------------------------------------------------

    print("\nLoading test dataset...")

    X_test, y_test = load_dataset(TEST_DIR)

    X_test, y_test = preprocess_dataset(
        X_test,
        y_test
    )

    print(f"Test samples : {len(X_test)}")

    # ----------------------------------------------------
    # Load Trained Model
    # ----------------------------------------------------

    print("\nLoading trained model...")

    model = load_model(MODEL_PATH)

    print("Model loaded successfully.")

    # ----------------------------------------------------
    # Evaluate
    # ----------------------------------------------------

    accuracy = evaluate_model(
        model,
        X_test,
        y_test
    )

    print(f"\nFinal Accuracy : {accuracy:.4f}")


if __name__ == "__main__":
    main()