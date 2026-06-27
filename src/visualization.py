"""
Visualization utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss curves
    and save them to disk.
    """

    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------
    # Accuracy
    # ---------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy",
        linewidth=2
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2
    )

    plt.title("Training vs Validation Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(plot_dir / "accuracy.png", dpi=300)

    plt.close()

    # ---------------------------------------------
    # Loss
    # ---------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
        linewidth=2
    )

    plt.title("Training vs Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(plot_dir / "loss.png", dpi=300)

    plt.close()

    print(f"\nPlots saved to: {plot_dir}")