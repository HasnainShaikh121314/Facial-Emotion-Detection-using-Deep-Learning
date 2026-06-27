"""
Evaluation metrics for Facial Expression Recognition.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.config import OUTPUT_DIR, CLASS_NAMES


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.

    Parameters
    ----------
    model : tensorflow.keras.Model

    X_test : ndarray

    y_test : ndarray

    Returns
    -------
    accuracy : float
    """

    print("\nEvaluating model...\n")

    # ----------------------------------------------------
    # Predictions
    # ----------------------------------------------------

    y_pred_prob = model.predict(X_test, verbose=0)

    y_pred = np.argmax(y_pred_prob, axis=1)

    y_true = np.argmax(y_test, axis=1)

    # ----------------------------------------------------
    # Accuracy
    # ----------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)

    print(f"Accuracy : {accuracy:.4f}")

    # ----------------------------------------------------
    # Classification Report
    # ----------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    )

    print("\nClassification Report\n")

    print(report)

    # ----------------------------------------------------
    # Save Report
    # ----------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    # ----------------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    plt.title("Confusion Matrix")

    plt.xticks(rotation=45)

    plt.tight_layout()

    cm_path = OUTPUT_DIR / "confusion_matrix.png"

    plt.savefig(
        cm_path,
        dpi=300
    )

    plt.close()

    print(f"\nConfusion Matrix saved to {cm_path}")

    return accuracy