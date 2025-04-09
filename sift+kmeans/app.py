import streamlit as st
import joblib
import cv2
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import tempfile

# ---- Load saved models ----
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_transform.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')
cluster_class_mapping = joblib.load('cluster_class_mapping.pkl')  # Load directly

# ---- Set max descriptor length used during training ----
max_length = 100
sift = cv2.SIFT_create(nfeatures=100)

# ---- Prediction function ----
def predict_class_from_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    if descriptors is None:
        return "No features found."

    padded_descriptor = pad_sequences([descriptors], maxlen=max_length, padding='post', truncating='post', dtype='float32')
    flat_descriptor = padded_descriptor.reshape((1, -1))
    scaled = scaler.transform(flat_descriptor)
    reduced = pca.transform(scaled)

    predicted_cluster = kmeans_model.predict(reduced)[0]
    return cluster_class_mapping.get(predicted_cluster, "Unknown")

# ---- Streamlit UI ----
st.title("Image Class Prediction")
st.write("Upload an image and get the predicted class based on SIFT + KMeans clustering.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL to OpenCV format
    image_np = np.array(image)
    
    # Predict class
    predicted_class = predict_class_from_image(image_np)
    st.success(f"Predicted Class: {predicted_class}")
