# test_model.py

import cv2
import pickle
from imutils import paths

def quantify_image(image, bins=(4, 6, 3)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def test_anomaly_detector(model_path, image_path, bins=(3, 3, 3)):
    model = pickle.loads(open(model_path, "rb").read())
    
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = quantify_image(hsv, bins)
    
    preds = model.predict([features])[0]
    label = "Anomaly" if preds == -1 else "Normal"
    return label, image

if __name__ == "__main__":
    model_path = "anomaly_detector.model"  # Path to the trained model
    image_path = "examples/test_image1.jpg"  # Path to the test image
    
    label, image = test_anomaly_detector(model_path, image_path)
    print(f"[INFO] Prediction: {label}")