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
    """
    Displays performance metrics and evaluation results for machine learning models.
    """

    version = 'v1'
    version_2 = 'v2'
    version_3 = 'v3'

    st.write("### Train, Validation and Test Set: Labels Frequencies")

    st.info(
        f"The dataset contains 1532 images. Including Healthy, Powdery, and Rust apple leaves.\n\n"
        f"The dataset was divided into 3 sets:\n\n Train Set - 70% "
        f"of the dataset.\n\n Validation Set - 10% of the dataset.\n\n Test "
        f"Set - 20% of the dataset.")

    labels_distribution = plt.imread(f"jupyter_notebooks/outputs/{version}/labels_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")

    st.success(
        f"The following plots show the model training accuracy and losses. "
        f"The accuracy is the measure of the model's prediction accuracy "
        f"compared to the true data (val_acc). The loss indicates incorrect "
        f"predictions on the train set (loss) and validation set (val_loss).")

    st.write("### Model History")
    col1, col2 = st.beta_columns(2)
    with col1: 
        model_acc = plt.imread(f"jupyter_notebooks/outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"jupyter_notebooks/outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')

    st.success(
        f"Both plots suggests the model exhibits a normal fit with no severe "
        f"overfitting or underfitting as the two lines follow the same "
        f"pattern.")

    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.info(
        f"The following data shows the model loss and accuracy on the test "
        f"dataset.")

    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))

    st.success(
        f"The prediction accuracy of the test set data is above 95%. This is "
        f"below 100%, suggesting the model is not overfitting.")
    
    st.write("---")
    
    st.markdown(
    f"<div style='font-size:16px; border: 2px solid #000000; padding: 10px; background-color: #FFE4B5;'>"
    f"The following plot shows the confusion matrix for the test dataset."
    f"<br><br>It shows the nine possible combinations of outcomes:<br><br>"
    f"True Positive / Negative - The model prediction is correct (dark Purple)"
    f"<br>False Positive / Negative - The model prediction is incorrect (light Blue)"
    f"<br>A good model has a high True rate and a low False rate."
    f"</div>", 
    unsafe_allow_html=True
    )

    st.write("### Confusion Matrix")
    col1 = st.beta_columns(1)[0]
    with col1:
        model_acc = plt.imread(f"jupyter_notebooks/outputs/{version_3}/confusion_matrix.png")
        st.image(model_acc, caption='Confusion Matrix')

    st.markdown(
    f"<div style='font-size:16px; border: 2px solid #000000; padding: 10px; background-color: #FFE4B5;'>"
    f"The confusion matrix shows the model made six incorrect predictions for Healthy class "
    f"when evaluating the test dataset. For leaves infected with powdery or rust, the model made four incorrect predictions."
    f"</div>", 
    unsafe_allow_html=True
    )

    st.write("---")

    # Load confusion matrix from joblib
    confusion_matrix = joblib.load(f"jupyter_notebooks/outputs/{version_3}/confusion_matrix.joblib")

    # Convert confusion matrix to a pandas DataFrame
    confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=['Predicted Healthy', 'Predicted Powdery', 'Predicted Rust'], index=['Actual Healthy', 'Actual Powdery', 'Actual Rust'])

    # Display confusion matrix in a Streamlit table
    st.write("Confusion Matrix:")
    st.table(confusion_matrix_df)

    st.write("---")

    # Load metrics from joblib
    metrics = joblib.load(f"jupyter_notebooks/outputs/{version_3}/metrics.joblib")

    # Convert metrics to a pandas DataFrame
    metrics_df = pd.DataFrame(metrics, index=['Value'])

    st.markdown(
    f"<div style='font-size:16px; border: 2px solid #000000; padding: 10px; margin-bottom: 30px; background-color: #E6E6FA;'>"
    f"The model evaluation metrics indicate the performance of the trained model on the test dataset."
    f"</div>", 
    unsafe_allow_html=True
    )

    # Display metrics in a Streamlit table
    st.write("Model Evaluation Metrics:")
    st.table(metrics_df)

    st.markdown(
    f"<div style='font-size:16px; border: 2px solid #000000; padding: 10px; background-color: #E6E6FA;'>"
    f"TThe model achieves an accuracy of 95.47%, indicating that it correctly predicts the classes of the test samples with an average accuracy of 95.47% across all classes."
    f"<br>"
    f"The precision, recall, and F1-score are also high, indicating that the model performs well in terms of both "
    f"correctly identifying positive samples and minimizing false positives."
    f"<br>"
    f"The specificity of the model, which measures the ability to correctly identify negative samples, is also high"
    f"at 94.39%."
    f"</div>", 
    unsafe_allow_html=True
    )

    st.write("---")

    st.write("* ### Conclusions")

    st.info(
    f"The ML model/pipeline has successfully fulfilled the following business requirements:\n\n"
    f"1. **Business Requirement 1:**\n\n"
    f"- This requirement is satisfied as observed in the Plant image Visualizer page, where healthy and infected leaves are distinguished by their appearance. Powdery infected leaves are identified by white deposits on their surface. Rust infected leaves are shown by orang spots on the leaves\n\n"
    f"2. **Business Requirement 2:**\n\n"
    f"- This requirement is met through the Plant Disease Detector page, which accurately predicts whether a leaf from an uploaded image is healthy or infected with powdery mildew or rust, achieving a 95% accuracy rate.\n\n"
    f"- Plant treatments are recommended based on the types of leaf diseases detected.\n\n"
    f"3. **Business Requirement 3:**\n\n"
    f"- This requirement is fulfilled as the Plant Disease Detector page allows users to download a report containing predictions made on uploaded images."
    )
