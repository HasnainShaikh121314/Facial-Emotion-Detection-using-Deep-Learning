# Facial Emotion Detection using Deep Learning

A deep learning-based Facial Emotion Recognition (FER) system built with TensorFlow/Keras. The project classifies facial expressions into seven emotions using a Convolutional Neural Network (CNN) trained on grayscale facial images.

## Features

- Facial emotion classification into 7 classes
- Custom CNN architecture
- TensorFlow / Keras implementation
- Automatic model checkpointing
- Early stopping and learning rate scheduling
- Training history visualization
- Confusion matrix generation
- Classification report
- Single image prediction
- Clean and modular project structure

---

## Dataset

The project expects the dataset in the following format:

```text
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Each image is resized to **48×48** grayscale before training.

---

## Model Architecture

The CNN consists of:

- 3 Convolutional Blocks
- Batch Normalization
- Max Pooling
- Dropout Regularization
- Fully Connected Layer (512 neurons)
- Softmax Output Layer

### Data Augmentation

The training pipeline uses:

- Random Horizontal Flip
- Random Rotation
- Random Zoom

---

## Project Structure

```text
Facial-Emotion-Detection-using-Deep-Learning/

├── data/
│   ├── train/
│   └── test/
│
├── models/
│
├── outputs/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── trainer.py
│   ├── visualization.py
│   └── metrics.py
│
├── train.py
├── evaluate.py
├── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/HasnainShaikh121314/Facial-Emotion-Detection-using-Deep-Learning.git
```

Move into the project directory

```bash
cd Facial-Emotion-Detection-using-Deep-Learning
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Evaluation

```bash
python evaluate.py
```

---

## Predict a Single Image

```bash
python predict.py --image path/to/image.jpg
```

---

## Output

The project automatically saves:

- Best trained model
- Training log
- Accuracy curve
- Loss curve
- Confusion matrix
- Classification report

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## Author

**Hasnain Shaikh**

GitHub:
https://github.com/HasnainShaikh121314
>>>>>>> 853aea0 (Refactor facial emotion detection project)
