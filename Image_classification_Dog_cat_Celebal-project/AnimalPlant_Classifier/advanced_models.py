import os
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ✅ Paths
data_path = os.path.join("new", "test")
model_save_path = "models"
os.makedirs(model_save_path, exist_ok=True)

# ✅ Load Images
def load_data(path):
    X, y = [], []
    for label, class_name in enumerate(["cats", "dogs"]):
        class_folder = os.path.join(path, class_name)
        for img_file in os.listdir(class_folder)[:1000]:
            img_path = os.path.join(class_folder, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (255, 255)).flatten()
                X.append(img)
                y.append(label)
    return np.array(X), np.array(y)

print("📥 Loading training data...")
X, y = load_data(data_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Models
models = {
    "SVM": SVC(kernel='linear', probability=True),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# ✅ Train & Save
for name, model in models.items():
    print(f"\n🔍 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ {name} Accuracy: {acc*100:.2f}%")
    
    model_path = os.path.join(model_save_path, f"{name}_model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Saved {name} model to {model_path}")
