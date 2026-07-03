# SalesCodeAI – Screen vs Real Image Detection

## Overview

This project detects whether an input image is:

- 📷 A genuine real-world photograph
- 📱 A photograph of a digital screen (recaptured image)

The solution uses handcrafted computer vision features combined with ensemble machine learning models to provide fast and lightweight on-device inference.

---

## Features

- Image preprocessing
- Laplacian variance (sharpness)
- Edge density analysis
- Local Binary Patterns (LBP)
- Histogram of Oriented Gradients (HOG)
- FFT frequency-domain features
- Sobel gradient features
- HSV color statistics
- Brightness and contrast estimation
- Reflection ratio
- Image entropy

---

## Machine Learning Pipeline

1. Extract handcrafted computer vision features.
2. Normalize features using `StandardScaler`.
3. Train multiple ensemble models:
   - Random Forest
   - Extra Trees
4. Select the best-performing model using 5-fold Stratified Cross Validation.
5. Save the trained model for inference.

---

## Dataset

- Real Images: 46
- Screen Images: 50

Images were captured under different lighting conditions, camera angles, and distances to improve robustness.

---

## Evaluation

- Validation Method: **5-Fold Stratified Cross Validation**
- Best Model: **Random Forest**
- Cross-Validation Accuracy: **89.18%**

---

## Project Structure

```
SalesCodeAI/
│
├── dataset/
│   ├── real/
│   └── screen/
│
├── models/
│
├── outputs/
│
├── feature_extractor.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Prediction

```bash
python predict.py image.jpg
```

Example Output:

```
0.84
```

Where:

- Near **0** → Real Photograph
- Near **1** → Screen Photograph

---

## Future Improvements

- Larger and more diverse dataset
- Hyperparameter optimization
- Lightweight CNN feature fusion
- Mobile deployment
- Real-time webcam interface

---

## Tech Stack

- Python
- OpenCV
- NumPy
- Scikit-Learn
- Scikit-Image
- Matplotlib
