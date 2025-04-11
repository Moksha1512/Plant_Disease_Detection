import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
import gdown

@st.cache_resource
def load_trained_model():
    model_path = "vgg16_plant_disease_model.h5"
    file_id = "1Kns43o5nbU6RruFtm3TEn-CShKY4QWqT"
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    model = tf.keras.models.load_model(model_path)
    return model

model = load_trained_model()

label_dict = {
    0: 'Apple___Black_rot',
    1: 'Apple___healthy',
    2: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    3: 'Corn_(maize)___Common_rust_',
    4: 'Corn_(maize)___healthy',
    5: 'Grape___Black_rot',
    6: 'Grape___healthy',
    7: 'Potato___Early_blight',
    8: 'Potato___Late_blight',
    9: 'Potato___healthy',
    10: 'Tomato_Bacterial_spot',
    11: 'Tomato_Early_blight',
    12: 'Tomato_healthy',
    13: 'Tomato_Late_blight',
    14: 'Tomato_Leaf_Mold'
}

st.title("🌿 Plant Disease Classifier (Transfer Learning - VGG16)")
st.write("Upload a leaf image to detect the plant disease (15 classes supported).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = cv2.resize(img, (150, 150))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.markdown(f"### 🧪 Prediction: `{label_dict[predicted_class]}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}`")
