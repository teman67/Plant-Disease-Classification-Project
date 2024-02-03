import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image
from src.data_management import load_pkl_file


def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Plot prediction probability results
    """

    class_labels = ['Healthy', 'Powdery', 'Rust']
    prob_per_class = pd.DataFrame({
        'Diagnostic': class_labels,
        'Probability': [1 - pred_proba if label != pred_class else pred_proba for label in class_labels]
    })

    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y='Probability',
        range_y=[0, 1],
        width=600, height=300, template='seaborn')
    st.plotly_chart(fig)



def resize_input_image(img, version):
    """
    Reshape image to average image size
    """
    image_shape = load_pkl_file(file_path=f"jupyter_notebooks/outputs/{version}/image_shape.pkl")
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)
    
    # Apply the same rescaling as during training
    my_image = np.array(img_resized) / 255.0
    my_image = np.expand_dims(my_image, axis=0)

    print("Resized Image Shape:", my_image.shape)

    return my_image


def load_model_and_predict(my_image, version):
    """
    Load and perform ML prediction over live images
    """

    model = load_model(f"jupyter_notebooks/outputs/{version}/plant_disease_detector.h5")

    pred_proba = model.predict(my_image)[0]
    print("Raw Predictions:", pred_proba)

    pred_class_index = np.argmax(pred_proba)
    pred_class = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}[pred_class_index]

    if pred_class == 'Healthy':
        st.write("The predictive analysis indicates the sample image is healthy.")
    else:
        st.write(
            f"The predictive analysis indicates the sample image is "
            f"**{pred_class.lower()}** with plant disease.")

    return pred_proba, pred_class

