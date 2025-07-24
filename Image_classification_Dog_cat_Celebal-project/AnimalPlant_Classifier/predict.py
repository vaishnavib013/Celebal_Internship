import os
import cv2
import numpy as np
import pickle

# --- CONFIG ---
IMG_SIZE = (255, 255)
MODEL_PATH = os.path.join("models", "RandomForest_model.pkl")
TEST_DIR = "test_images"  # Folder containing test images directly

# --- LOAD MODEL ---
with open(MODEL_PATH, "rb") as f:
    model, label_encoder = pickle.load(f)

# --- PREDICT FUNCTION ---
def predict_image(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = gray.flatten().reshape(1, -1)
        pred = model.predict(features)
        label = label_encoder.inverse_transform(pred)
        return label[0]
    except Exception as e:
        print(f"❌ Error reading {img_path}: {e}")
        return None

# --- RUN PREDICTIONS ---
print("🔎 Predictions on images in 'test_images' folder:\n")

for img_file in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, img_file)
    if not os.path.isfile(img_path):
        continue
    predicted_label = predict_image(img_path)
    print(f"🖼️ {img_file} → Predicted: {predicted_label}")
