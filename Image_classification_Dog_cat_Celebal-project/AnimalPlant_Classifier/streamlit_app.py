import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image

# --- CONFIG ---
IMG_SIZE = (255, 255)
MODEL_PATH = "models/RandomForest_model.pkl"

# --- Load the Model ---
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model, label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model()

# --- Prediction Function ---
def predict_image(img: Image.Image):
    img = img.resize(IMG_SIZE)
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = gray.flatten().reshape(1, -1)
    pred = model.predict(features)
    label = label_encoder.inverse_transform(pred)
    return label[0]

# --- Streamlit UI ---
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("🐱 vs 🐶 Cat vs Dog Image Classifier")
st.write("Upload an image to predict whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        result = predict_image(img)

    st.success(f"✅ Prediction: It's a **{result}**!")
