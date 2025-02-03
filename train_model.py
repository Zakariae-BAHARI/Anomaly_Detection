# train_model.py

from imutils import paths
import numpy as np
import cv2
import pickle
from sklearn.ensemble import IsolationForest

def quantify_image(image, bins=(4, 6, 3)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def load_dataset(datasetPath, bins):
    imagePaths = list(paths.list_images(datasetPath))
    data = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = quantify_image(image, bins)
        data.append(features)
    return np.array(data)

def train_anomaly_detector(dataset_path, model_path, bins=(3, 3, 3)):
    print("[INFO] Preparing dataset...")
    data = load_dataset(dataset_path, bins)
    
    print("[INFO] Fitting anomaly detection model...")
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(data)
    
    with open(model_path, "wb") as f:
        f.write(pickle.dumps(model))
    print(f"[INFO] Model saved to {model_path}")

if __name__ == "__main__":
    dataset_path = "forest"  
    model_path = "anomaly_detector.model"  
    train_anomaly_detector(dataset_path, model_path)