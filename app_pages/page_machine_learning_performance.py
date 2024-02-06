import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import joblib


def page_machine_learning_performance_metrics():
    version = 'v1'
    version_2 = 'v2'
    version_3 = 'v3'

    st.write("### Train, Validation and Test Set: Labels Frequencies")

    labels_distribution = plt.imread(f"jupyter_notebooks/outputs/{version}/labels_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")


    st.write("### Model History")
    col1, col2 = st.beta_columns(2)
    with col1: 
        model_acc = plt.imread(f"jupyter_notebooks/outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"jupyter_notebooks/outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
    
    st.write("### Confusion Matrix")
    col1 = st.beta_columns(1)[0]
    with col1:
        model_acc = plt.imread(f"jupyter_notebooks/outputs/{version_3}/confusion_matrix.png")
        st.image(model_acc, caption='Confusion Matrix')
    st.write("---")

    # Load confusion matrix from joblib
    confusion_matrix = joblib.load(f"jupyter_notebooks/outputs/{version_3}/confusion_matrix.joblib")

    # Convert confusion matrix to a pandas DataFrame
    confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=['Predicted Healthy', 'Predicted Powdery', 'Predicted Rust'], index=['Actual Healthy', 'Actual Powdery', 'Actual Rust'])

    # Display confusion matrix in a Streamlit table
    st.write("Confusion Matrix:")
    st.table(confusion_matrix_df)

    # Load metrics from joblib
    metrics = joblib.load(f"jupyter_notebooks/outputs/{version_3}/metrics.joblib")

    # Convert metrics to a pandas DataFrame
    metrics_df = pd.DataFrame(metrics, index=['Value'])

    # Display metrics in a Streamlit table
    st.write("Model Evaluation Metrics:")
    st.table(metrics_df)