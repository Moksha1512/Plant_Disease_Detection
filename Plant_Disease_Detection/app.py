import streamlit as st
from PIL import Image
import torch
import joblib
from torchvision import transforms
from vit_model import ViTForClassfication  
from io import BytesIO
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


# Function to load the model and weights
def load_model(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Model weights loaded from {weights_path}")
    return model

# Function to load label encoder
def load_label_encoder(encoder_path):
    return joblib.load(encoder_path)

# Function to predict the class of an image
def predict_image(image, model, label_encoder, device):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to match model's expected input size
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalization parameters
    ])

    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image.to(device))

    logits = output[0] if isinstance(output, tuple) else output
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    _, predicted_class_idx = torch.max(probabilities, 1)
    
    predicted_label = label_encoder.inverse_transform(predicted_class_idx.cpu().numpy())
    return predicted_label[0]

# Streamlit app layout
st.title("Plant Disease Classification")
st.write("Upload an image of a plant leaf to classify its disease.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Load model and label encoder (this will be done only once)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTForClassfication(config)  # Replace with your model's config
    weights_path = "plant_disease_vit_checkpoint_epoch_26.pt"  # Path to model weights
    encoder_path = "label_encoder.pkl"  # Path to label encoder
    
    model = load_model(model, weights_path, device)
    label_encoder = load_label_encoder(encoder_path)

    # Make a prediction
    if st.button('Classify'):
        predicted_label = predict_image(image, model, label_encoder, device)
        st.write(f"Predicted Disease: {predicted_label}")
