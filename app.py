import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import torch
import joblib
import tensorflow as tf
import numpy as np
import os
from torchvision import transforms
import cv2
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==== Imports from folders ====
from Plant_Disease_Detection.vit_model import ViTForClassfication
from cv_streamlit.variational_autoencoder import VariationalAutoEncoder
from cv_streamlit.classifier import Classifier

# ==== Common Settings ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(page_title="Plant Disease Detection App", layout="wide")

# ==== Sidebar Navigation ====
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["ViT Model", "VAE Model", "CNN+Attention Model", "SIFT + GMM Model"],
        icons=["activity", "cpu", "image", "bounding-box"],
        default_index=0,
    )

# ==== ViT Model Page ====
if selected == "ViT Model":
    st.title("🌿 Plant Disease Classification (ViT Model)")

    config = {
        "patch_size": 4,
        "hidden_size": 96,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "intermediate_size": 4 * 96,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "image_size": 64,
        "num_classes": 15,
        "num_channels": 3,
        "qkv_bias": True,
        "use_faster_attention": True,
    }

    @st.cache_resource
    def load_vit_model(weights_path, encoder_path):
        model = ViTForClassfication(config)
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        label_encoder = joblib.load(encoder_path)
        return model, label_encoder

    vit_model, vit_encoder = load_vit_model(
        "Plant_Disease_Detection/plant_disease_vit_checkpoint_epoch_26.pt",
        "Plant_Disease_Detection/label_encoder.pkl"
    )

    uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Classify with ViT"):
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = vit_model(input_tensor)
                logits = output[0] if isinstance(output, tuple) else output
                probs = torch.nn.functional.softmax(logits, dim=1)
                _, pred_idx = torch.max(probs, 1)
                label = vit_encoder.inverse_transform(pred_idx.cpu().numpy())[0]
            st.success(f"🧠 Predicted Disease: **{label}**")

# ==== CNN + Attention Page ====
elif selected == "CNN+Attention Model":
    st.title("🌿 Plant Disease Classification (CNN + Attention)")

    CLASS_LABELS = [
        'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
        'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
        'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
        'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
    ]

    IMAGE_SIZE = (128, 128)

    @st.cache_resource
    def load_tf_model():
        model = tf.keras.models.load_model("CNN_ATTENTION/plant_disease_model.h5", compile=False)
        return model

    cnn_model = load_tf_model()

    uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"], key="cnn_upload")

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Leaf Image', use_container_width=True)

        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = cnn_model.predict(img_array)
        predicted_class = CLASS_LABELS[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.markdown(f"### ✅ Prediction: **{predicted_class}**")
        st.markdown(f"**Confidence:** {confidence:.2%}")

# ==== VAE Model Page ====
elif selected == "VAE Model":
    st.title("🌿 Plant Disease Classification (VAE-based Model)")

    class_names = [
        'Tomato__Target_Spot', 'Tomato_Early_blight', 'Tomato_Leaf_Mold',
        'Tomato_Bacterial_spot', 'Potato___Early_blight',
        'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Septoria_leaf_spot',
        'Tomato_healthy', 'Potato___healthy', 'Tomato__Tomato_mosaic_virus',
        'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Pepper__bell___healthy',
        'Potato___Late_blight', 'Pepper__bell___Bacterial_spot', 'Tomato_Late_blight'
    ]

    @st.cache_resource
    def load_vae_models():
        vae = VariationalAutoEncoder(in_channels=3).to(device)
        clf = Classifier(in_features=30).to(device)
        vae.load_state_dict(torch.load("cv_streamlit/vae_weights.pth", map_location=device))
        clf.load_state_dict(torch.load("cv_streamlit/classifier_vae_weights.pth", map_location=device))
        vae.eval()
        clf.eval()
        return vae, clf

    vae, classifier = load_vae_models()

    uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"], key="vae_upload")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Classify with VAE"):
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                encoded, _, _, _ = vae(input_tensor)
                output = classifier(encoded)
                pred_class = torch.argmax(output, dim=1).item()
                st.success(f"🧠 Predicted Disease: **{class_names[pred_class]}**")
                
   # ==== SIFT + GMM Model Page ====
elif selected == "SIFT + GMM Model":
    st.title("🧠 SIFT + GMM Image Classification")

    @st.cache_resource
    def load_sift_gmm():
        scaler = joblib.load('sift+kmeans/scaler.pkl')
        pca = joblib.load('sift+kmeans/pca_transform.pkl')
        gmm_model = joblib.load('sift+kmeans/gmm_model.pkl')
        cluster_class_mapping = joblib.load('sift+kmeans/cluster_class_mapping.pkl')
        return scaler, pca, gmm_model, cluster_class_mapping

    scaler, pca, gmm_model, cluster_class_mapping = load_sift_gmm()
    sift = cv2.SIFT_create(nfeatures=100)
    max_length = 100

    def predict_class_from_image(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        if descriptors is None:
            return "No features found."

        padded = pad_sequences([descriptors], maxlen=max_length, padding='post', truncating='post', dtype='float32')
        flat = padded.reshape((1, -1))
        scaled = scaler.transform(flat)
        reduced = pca.transform(scaled)
        predicted_cluster = gmm_model.predict(reduced)[0]
        return cluster_class_mapping.get(predicted_cluster, "Unknown")

    uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"], key="sift_gmm_upload")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        image_np = np.array(image)

        if st.button("🔍 Classify with GMM"):
            pred = predict_class_from_image(image_np)
            st.success(f"🧠 Predicted Class: **{pred}**")
