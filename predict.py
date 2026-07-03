import sys
import joblib
import numpy as np

from feature_extractor import extract_features

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"


def predict(image_path):

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features = extract_features(image_path)

    features = scaler.transform([features])

    probability = model.predict_proba(features)[0][1]

    probability = float(np.clip(probability, 0, 1))

    return probability


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python predict.py image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    try:

        score = predict(image_path)

        print(round(score,4))
    except Exception as e:

        print(e)