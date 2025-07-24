# 🐶🐱 Animal Image Classifier – Cat vs Dog
## Business Understanding
Image classification plays a crucial role in several real-world applications, especially in automating identification processes that would otherwise require human effort. In this project, we focus on distinguishing between cats and dogs using traditional machine learning techniques, making it a foundational case study in image-based animal classification.

## 🎯 Business Objective
The primary objective of this project is to:

Automatically classify images into either "Cat" or "Dog" categories using a machine learning pipeline.

Develop a lightweight, fast, and interpretable model that can be used even without GPUs, making it suitable for low-resource environments such as basic mobile apps, edge devices, or educational platforms.

Provide a simple user interface via Streamlit for uploading images and viewing predictions in real-time.

Serve as a baseline system for further extending into multi-class animal classification (like horse, elephant, etc.) or plant disease detection by simply retraining the model with a new dataset.

## ⚙️ Technologies Used

- Python
- Jupyter Notebook / Google Colab
- Scikit-learn
- OpenCV
- Numpy, Pandas
- Matplotlib
- Streamlit
- TensorFlow (for CNN - optional part)
- Git, GitHub


## 🗂 Project Structure

```bash
AnimalPlant_Classifier/
│
├── data/                         # Raw image folders
│   ├── Animal/
│       ├── cat/
│       └── dog/
├── models/                      # Saved .pkl model files
├── test_images/                 # Images used for testing/prediction
│
├── main_classifier.py           # Contains DT, RF, KNN, GNB
├── advanced_models.py           # Contains SVM, Logistic Regression
├── predict.py                   # Predicts class of new image
├── streamlit_app.py             # Streamlit Web App
├── README.md
└── requirements.txt             # All dependencies


Models and Accuracies
| Model                    | Accuracy                    |
| ------------------------ | --------------------------- |
| Logistic Regression      | 51.50%                      |
| SVM                      | 53.50%                      |
| K-Nearest Neighbors      | 57.00%                      |
| Decision Tree Classifier | 54.00%                      |
| Gaussian Naive Bayes     | 56.00%                      |
| Random Forest Classifier | **62.00%** ✅                |
| CNN using TensorFlow     | 90.00% (for reference only) |


📌 We selected Random Forest Classifier for our deployed model as it had the best performance among traditional ML models.

# Feature Extraction
HOG (Histogram of Oriented Gradients) was used for extracting features from grayscale images.

All images were resized to 255X255 *3 and flattened into vectors after HOG feature extraction.

Prediction Script
predict.py loads each trained model (from .pkl files) and runs prediction on images from the test_images/ folder. Outputs are printed on the console.

🧪 Model Training
main_classifier.py:
Trained: KNN, DecisionTree, RandomForest, GaussianNB

advanced_models.py:
Trained: SVM, Logistic Regression

Each model was saved using:

python
Copy code
with open('models/model_name.pkl', 'wb') as f:
    pickle.dump(model, f)

Deployed 