# app.py

import streamlit as st
import cv2
import pickle
import numpy as np
from imutils import paths
from matplotlib import pyplot as plt

# Fonction pour charger le modèle
def load_model(model_path):
    return pickle.loads(open(model_path, "rb").read())

# Fonction pour quantifier une image
def quantify_image(image, bins=(4, 6, 3)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Fonction pour détecter les anomalies
def detect_anomaly(model, image_path, bins=(3, 3, 3)):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = quantify_image(hsv, bins)
    preds = model.predict([features])[0]
    label = "Anomaly" if preds == -1 else "Normal"
    return label, image

# Interface utilisateur avec Streamlit
def main():
    st.title("Anomaly Detection App")
    
    # Charger le modèle
    model_path = "anomaly_detector.model"
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
    except FileNotFoundError:
        st.error("Model not found. Please train the model first.")
        return
    
    # Upload d'une image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Enregistrer l'image temporairement
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.read())
        
        # Détecter les anomalies
        label, image = detect_anomaly(model, "temp_image.jpg")
        
        # Afficher les résultats
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Prediction: {label}", use_column_width=True)
        st.write(f"**Prediction:** {label}")

if __name__ == "__main__":
    main()