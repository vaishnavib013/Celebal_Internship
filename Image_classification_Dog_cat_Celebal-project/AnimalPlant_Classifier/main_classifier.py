import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

# --- CONFIG ---
IMG_SIZE = (255, 255)
DATA_PATH = os.path.join('new', 'test')  # Using test as the main dataset

# --- LOAD DATA ---
def load_data():
    X, y = [], []
    for label in os.listdir(DATA_PATH):  # cats, dogs
        folder = os.path.join(DATA_PATH, label)
        for img_file in os.listdir(folder):
            try:
                img_path = os.path.join(folder, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, IMG_SIZE)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                X.append(gray.flatten())
                y.append(label)
            except Exception as e:
                print(f"Error reading {img_file}: {e}")
    return np.array(X), np.array(y)

print("📥 Loading data...")
X, y = load_data()

# --- SPLIT ---
print("🔀 Splitting into train and test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- ENCODE LABELS ---
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# --- MODELS ---
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(),
    "GaussianNB": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
   # "SVM": SVC(kernel='linear', probability=True)
}

accuracies = {}

# --- TRAIN AND SAVE ---
os.makedirs("models", exist_ok=True)

for name, model in models.items():
    print(f"🧠 Training {name}...")
    model.fit(X_train, y_train_encoded)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_encoded, y_pred)
    accuracies[name] = acc
    print(f"✅ {name} Accuracy: {acc:.4f}")
    model_path = os.path.join("models", f"{name.replace(' ', '')}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump((model, le), f)

# --- BEST MODEL ---
best_model = max(accuracies, key=accuracies.get)
print(f"\n🏆 Best model: {best_model} with Accuracy: {accuracies[best_model]:.4f}")
