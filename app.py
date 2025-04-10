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
import torchvision.models as models
import torch.nn as nn
import gdown

# ==== Imports from folders ====
from VIT.vit_model import ViTForClassfication
from  (VAE+MLP).variational_autoencoder import VariationalAutoEncoder
from (VAE+MLP).classifier import Classifier

# ==== Common Settings ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(page_title="Plant Disease Detection App", layout="wide")

# ==== Sidebar Navigation ====
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["ViT Model", "VAE Model", "CNN+Attention Model", "SIFT + GMM Model", "ResNet Model"],
        icons=["activity", "cpu", "image", "bounding-box", "database"],
        default_index=0,
    )

# ==== Helper: Unified image uploader ====
def upload_image(key=None):
    file = st.file_uploader("\U0001F4F7 Upload an image", type=["jpg", "jpeg", "png"], key=key)
    if file:
        image = Image.open(file).convert("RGB")
        image = image.resize((256, 256))  # Reduce image size for efficiency
        st.image(image, caption="Uploaded Image", use_container_width=False, width=300)  # Reduced display size
        return image
    return None

# ==== ViT Model Page ====
if selected == "ViT Model":
    st.title("\U0001F33F Plant Disease Classification (ViT Model)")

    config = {
        "patch_size": 4, "hidden_size": 96, "num_hidden_layers": 8,
        "num_attention_heads": 8, "intermediate_size": 384,
        "hidden_dropout_prob": 0.1, "attention_probs_dropout_prob": 0.1,
        "initializer_range": 0.02, "image_size": 64,
        "num_classes": 15, "num_channels": 3, "qkv_bias": True,
        "use_faster_attention": True,
    }

    @st.cache_resource
    def load_vit_model():
        model = ViTForClassfication(config)
        checkpoint = torch.load("VIT/plant_disease_vit_checkpoint_epoch_26.pt", map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        model.to(device).eval()
        encoder = joblib.load("VIT/label_encoder.pkl")
        return model, encoder

    with st.spinner("Loading ViT model..."):
        vit_model, vit_encoder = load_vit_model()

    image = upload_image("vit_upload")
    if image:
        transform = transforms.Compose([
            transforms.Resize((64, 64)), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = vit_model(input_tensor)[0]
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, 1).item()
            label = vit_encoder.inverse_transform([pred_idx])[0]
        st.success(f"\U0001F9E0 **Predicted Disease**: {label}")

# ==== CNN + Attention Page ====
elif selected == "CNN+Attention Model":
    st.title("\U0001F33F Plant Disease Classification (CNN + Attention)")

    CLASS_LABELS = [
        'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
        'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
        'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
        'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
    ]

    @st.cache_resource
    def load_tf_model():
        return tf.keras.models.load_model("CNN_ATTENTION/plant_disease_model.h5", compile=False)

    with st.spinner("Loading CNN + Attention model..."):
        cnn_model = load_tf_model()

    image = upload_image("cnn_upload")
    if image:
        img = image.resize((128, 128))
        input_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        prediction = cnn_model.predict(input_array)
        pred_class = CLASS_LABELS[np.argmax(prediction)]
        confidence = np.max(prediction)
        st.markdown(f"### ✅ **Prediction**: {pred_class}")
        st.markdown(f"**Confidence**: {confidence:.2%}")

# ==== VAE Model Page ====
elif selected == "VAE Model":
    st.title("\U0001F33F Plant Disease Classification (VAE-based Model)")

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
        vae.load_state_dict(torch.load("VAE+MLP/vae_weights.pth", map_location=device))
        clf.load_state_dict(torch.load("VAE+MLP/classifier_vae_weights.pth", map_location=device))
        vae.eval(); clf.eval()
        return vae, clf

    with st.spinner("Loading VAE + Classifier..."):
        vae, classifier = load_vae_models()

    image = upload_image("vae_upload")
    if image:
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            encoded, _, _, _ = vae(input_tensor)
            logits = classifier(encoded)
            pred_idx = torch.argmax(logits, dim=1).item()
            st.success(f"\U0001F9E0 **Predicted Disease**: {class_names[pred_idx]}")

# ==== SIFT + GMM Model Page ====
elif selected == "SIFT + GMM Model":
    st.title("\U0001F33F Plant Disease Classification (SIFT + GMM)")

    @st.cache_resource
    def load_sift_gmm():
        return (
            joblib.load('sift+kmeans/scaler.pkl'),
            joblib.load('sift+kmeans/pca_transform.pkl'),
            joblib.load('sift+kmeans/gmm_model.pkl'),
            joblib.load('sift+kmeans/cluster_class_mapping.pkl')
        )

    with st.spinner("Loading SIFT + GMM components..."):
        scaler, pca, gmm_model, cluster_class_mapping = load_sift_gmm()
        sift = cv2.SIFT_create(nfeatures=100)

    def predict_gmm(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kp, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            return "No features found."
        padded = pad_sequences([descriptors], maxlen=100, padding='post', truncating='post', dtype='float32')
        reduced = pca.transform(scaler.transform(padded.reshape(1, -1)))
        cluster = gmm_model.predict(reduced)[0]
        return cluster_class_mapping.get(cluster, "Unknown")

    image = upload_image("sift_gmm_upload")
    if image:
        pred = predict_gmm(np.array(image))
        st.success(f"\U0001F9E0 **Predicted Class**: {pred}")

# ==== ResNet Model Page ====
elif selected == "ResNet Model":
    st.title("\U0001F33F Plant Disease Classification (ResNet Model)")

    @st.cache_resource
    def load_resnet():
        url = "https://drive.google.com/uc?id=1DEtyePl-vgjj-qvZ_xeRz0jXipqoX35W"
        output = "plant_disease_resnet50.pth"
        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)

        labels = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
           'Potato___Early_blight', 'Potato___Late_blight',
           'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
           'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
           'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
           'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
           'Tomato_healthy']

        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.01), nn.Linear(model.fc.in_features, len(labels)))
        model.load_state_dict(torch.load(output, map_location=device))
        model.to(device).eval()
        return model, labels

    with st.spinner("Loading ResNet model..."):
        resnet_model, resnet_labels = load_resnet()

    image = upload_image("resnet_upload")
    if image:
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = resnet_model(input_tensor)
            pred = torch.argmax(output, 1).item()
            st.success(f"\U0001F9E0 **Predicted Disease**: {resnet_labels[pred]}")
