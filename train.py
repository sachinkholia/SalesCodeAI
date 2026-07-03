import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from feature_extractor import extract_features

REAL_DIR = "dataset/real"
SCREEN_DIR = "dataset/screen"

X = []
y = []

print("Extracting REAL images...")

for file in tqdm(os.listdir(REAL_DIR)):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    try:
        X.append(extract_features(os.path.join(REAL_DIR, file)))
        y.append(0)
    except Exception as e:
        print(e)

print("Extracting SCREEN images...")

for file in tqdm(os.listdir(SCREEN_DIR)):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    try:
        X.append(extract_features(os.path.join(SCREEN_DIR, file)))
        y.append(1)
    except Exception as e:
        print(e)

X=np.array(X)
y=np.array(y)

print("Dataset:",X.shape)

scaler=StandardScaler()

X=scaler.fit_transform(X)

models={
    "RandomForest": RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
}

best_acc=0
best_model=None
best_name=""

cv=StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name,model in models.items():

    score=cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    acc=np.mean(score)

    print(name,acc)

    if acc>best_acc:

        best_acc=acc

        best_model=model

        best_name=name

best_model.fit(X,y)

joblib.dump(best_model,"models/best_model.pkl")
joblib.dump(scaler,"models/scaler.pkl")

pred=best_model.predict(X)

cm=confusion_matrix(y,pred)

disp=ConfusionMatrixDisplay(cm)

disp.plot()

plt.savefig("outputs/confusion_matrix.png")

print()

print("="*50)
print("Best Model :",best_name)
print("Cross Validation Accuracy :",best_acc*100)
print("="*50)