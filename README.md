# Anomaly Detection
This project is about finding unusual or unexpected images (anomalies) using Python, OpenCV, and scikit-learn. It is a computer vision project that uses a machine learning method called "Isolation Forest" to separate normal images from anomalies. The model works by analyzing patterns in the colors of the images.

## Overview
The program processes images and calculates their color distribution. It trains a model using these features to identify unusual images. Once trained, you can test the model with new images to classify them as either "normal" or "anomalies."

To facilitate the use of the model, we have developed a user-friendly interface using Streamlit , making it easy for users to interact with the application and test new images.

## Data Description
The project uses images as input data:

Training Data : The model learns from a folder of images that are considered "normal."
Testing Data : You can test the model with new images to see if they are "normal" or "anomalies."

## Example:
Training images : Pictures of healthy forests.
Test images : May include healthy forests and some with anomalies like fires or disease.
You can replace the sample data with your own images by placing them in the appropriate folders.

## Features
Computer Vision : Uses color histograms to quantify image features.
Machine Learning : Trains an Isolation Forest model to detect anomalies.
User Interface : A web-based interface built with Streamlit for easy interaction.

## How to Use
Train the Model : Train the model on a dataset of normal images.
Test the Model : Test the model with new images to classify them as "normal" or "anomalie."

## Technologies Used
Python : Programming language used for development.
OpenCV : Library for image processing and feature extraction.
scikit-learn : Machine learning library for training the Isolation Forest model.
Streamlit : Framework for creating the user interface.
