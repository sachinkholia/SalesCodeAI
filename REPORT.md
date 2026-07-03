# SalesCodeAI Assignment Report

## Objective

Develop a lightweight computer vision system to distinguish between genuine photographs and photographs of digital screens.

## Approach

The solution uses handcrafted computer vision features instead of training a deep deep-learning model from scratch. Features extracted include:

- Laplacian Variance (sharpness)
- Edge Density
- Histogram of Oriented Gradients (HOG)
- Local Binary Patterns (LBP)
- FFT Frequency Features
- Sobel Gradient Features
- HSV Color Statistics
- Brightness and Contrast
- Reflection Ratio
- Image Entropy

The extracted feature vectors are normalized using `StandardScaler` and evaluated using ensemble learning models. Both **Random Forest** and **Extra Trees** classifiers were trained, and the best-performing model was selected using **5-fold Stratified Cross Validation**.

## Dataset

- Real Images: 46
- Screen Images: 50

Images were captured under different lighting conditions, camera angles, and distances.

## Results

- **Best Model:** Random Forest
- **Validation:** 5-Fold Stratified Cross Validation
- **Cross-Validation Accuracy:** **89.18%**
- **Inference Latency:** ~10 ms per image on CPU
- **Inference Cost:** ~$0 per image (fully on-device)

## Future Improvements

- Collect a larger and more diverse dataset.
- Perform hyperparameter optimization.
- Combine handcrafted features with a lightweight CNN feature extractor.
- Deploy as a real-time mobile or web application.
