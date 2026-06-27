"""
Project Configuration
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

MODEL_NAME = "best_model.h5"
MODEL_PATH = MODEL_DIR / MODEL_NAME

# =============================================================================
# Image Parameters
# =============================================================================

IMAGE_SIZE = (48, 48)

CHANNELS = 1

# =============================================================================
# Model Parameters
# =============================================================================

NUM_CLASSES = 7

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# =============================================================================
# Training Parameters
# =============================================================================

BATCH_SIZE = 64

EPOCHS = 50

LEARNING_RATE = 0.0005

RANDOM_STATE = 42