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
    0: 'Pepper_bell_Bacterial_spot',
    1: 'Pepperbell__healthy',
    2: 'Potato__Early_blight',
    3: 'Potato__Late_blight',
    4: 'Potato___healthy',
    5: 'Tomato_Bacterial_spot',
    6: 'Tomato_Early_blight',
    7: 'Tomato_Late_blight',
    8: 'Tomato_Leaf_Mold',
    9: 'Tomato_Septoria_leaf_spot',
    10: 'Tomato_Spider_mites_Two_spotted_spider_mite',
    11: 'Tomato_Target_Spot',
    12: 'TomatoTomato_YellowLeaf_Curl_Virus',
    13: 'Tomato__Tomato_mosaic_virus',
    14: 'Tomato_healthy'
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
