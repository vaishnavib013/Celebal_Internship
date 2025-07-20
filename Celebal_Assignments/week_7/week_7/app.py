import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_iris

model = joblib.load("isris.joblib")
iris = load_iris()
target_names = iris.target_names

st.title("Iris Species Prediction App (using pre-trained KNN model)")
st.write("This app predicts the species of an Iris flower using a trained KNN model.")

st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width  = st.sidebar.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width  = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)

st.subheader("Prediction Result")
st.write(f"**Predicted Species:** {target_names[prediction]}")
st.write("Prediction Probabilities:")
proba_df = pd.DataFrame(prediction_proba, columns=target_names)
st.bar_chart(proba_df.T)

if st.checkbox("Show Input Values"):
    st.write(pd.DataFrame(input_data, columns=iris.feature_names))
