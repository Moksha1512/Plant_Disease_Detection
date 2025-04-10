import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from variational_autoencoder import VariationalAutoEncoder
from classifier import Classifier

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"🔌 Device in use: `{device}`")

# Class labels
class_name = [
    'Tomato__Target_Spot', 'Tomato_Early_blight', 'Tomato_Leaf_Mold',
    'Tomato_Bacterial_spot', 'Potato___Early_blight',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Septoria_leaf_spot',
    'Tomato_healthy', 'Potato___healthy', 'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Pepper__bell___healthy',
    'Potato___Late_blight', 'Pepper__bell___Bacterial_spot', 'Tomato_Late_blight'
]

# Initialize models
vae = VariationalAutoEncoder(in_channels=3).to(device)
clf = Classifier(in_features=30).to(device)

# Load weights
vae.load_state_dict(torch.load('cv_streamlit/vae_weights.pth', map_location=device))
clf.load_state_dict(torch.load('cv_streamlit/classifier_vae_weights.pth', map_location=device))

# Set models to eval mode
vae.eval()
clf.eval()

# Streamlit UI
st.title("🌿 Plant Disease Classifier (VAE-based)")
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Encode and classify
    with torch.no_grad():
        encoded, _, _, _ = vae(input_tensor)
        output = clf(encoded)
        pred_class = torch.argmax(output, dim=1).item()

    st.success(f"🧠 Predicted class: **{class_name[pred_class]}**")
