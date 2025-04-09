import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set constants
IMAGE_SIZE = (128, 128)
CLASS_LABELS = [
     'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
     'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
     'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_model.h5", compile=False)
    return model

model = load_model()

# Streamlit app UI
st.title("🌿 Plant Disease Classifier")
st.write("Upload a leaf image to detect plant disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Leaf Image', use_column_width=True)

    # Preprocessing
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### ✅ Prediction: **{predicted_class}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")
