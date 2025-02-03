# Anomaly Detection
This project is about finding unusual or unexpected images (anomalies) using Python, OpenCV, and scikit-learn. We use a machine learning method called "Isolation Forest" to separate normal images from anomalies. It works by looking at patterns in the colors of the images.

## Overview
The program processes images and calculates their color distribution. It trains a model using these features to identify unusual images. You can test the trained model with new images to see if they are normal or anomalies.

## Data Description
The project uses images as input data:

Training Data: The model learns from a folder of images that are considered "normal." 
Testing Data: You can test the model with new images to see if they are "normal" or "anomalies." 

For example:

Training images could be pictures of healthy forests. 
Test images might include healthy forests and some with anomalies like fires or disease. 
You can replace the sample data with your own images by putting them in the appropriate folders.

## How to Use
Train the model on a dataset of normal images. Test the model with new images to classify them as "normal" or "anomalie."
