import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import requests
from io import BytesIO
import os

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities
)

def page_plant_disease_detector_body():
    st.info(
        f"* The client is interested in telling whether a given plant image is healthy or not. Besides, if it is not healthy, which type of plant disease it has."
    )

    st.write(
        f"* You can download a set of healthy, powdery and rust images for live prediction. "
        f"You can download the images from [here](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset?select=Test)."
    )

    st.write("---")

    # Option to upload image files
    images_buffer = st.file_uploader('Upload plant image(s) or provide URL(s). You may select more than one.',
                                     type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    # Option to enter image URLs
    st.write("Enter image URL(s) (one URL per line):")
    image_urls = st.text_area("")

    # Button to confirm URLs for image prediction
    confirm_button = st.button("Confirm URLs")

    # Combine uploaded images and images from URLs
    images_from_urls = []
    image_filenames = []
    if confirm_button:
        if image_urls:
            urls = image_urls.split("\n")
            for url in urls:
                try:
                    response = requests.get(url)
                    img_pil = Image.open(BytesIO(response.content))
                    images_from_urls.append(img_pil)
                    # Extract filename from URL
                    image_filenames.append(os.path.basename(url))
                except Exception as e:
                    st.warning(f"Unable to fetch image from URL: {url}")

    if images_buffer is not None:
        images_from_buffer = [Image.open(image) for image in images_buffer]
        # Extract filenames from buffer
        buffer_filenames = [image.name for image in images_buffer]
        image_filenames.extend(buffer_filenames)
    else:
        images_from_buffer = []

    images_to_process = images_from_buffer + images_from_urls

    # Define a dictionary to map class indices to class names
    class_mapping = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

    if images_to_process:
        df_report = pd.DataFrame([])
        for img_pil, filename in zip(images_to_process, image_filenames):

            st.info(f"Plant Image - {filename}")
            st.image(img_pil, caption=f"Image Size: {img_pil.size[0]}px width x {img_pil.size[1]}px height")

            version = 'v3'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)

            # Convert numpy.float32 values to Python floats
            pred_proba = [float(value) for value in pred_proba]

            # Print raw predictions with class names
            class_names = [class_mapping[i] for i in range(len(pred_proba))]
            raw_predictions_table = pd.DataFrame(list(zip(class_names, pred_proba)), columns=['Class', 'Prediction Probability'])
            st.table(raw_predictions_table)

            # Create a pie chart to display predictions
            fig = px.pie(raw_predictions_table, names='Class', values='Prediction Probability', title='Prediction Probabilities')
            st.plotly_chart(fig)

            # Print the predicted class with class name
            predicted_class_name = class_mapping[np.argmax(pred_proba)]
            
            df_report = df_report.append({"Name": filename, 'Result': predicted_class_name},
                                         ignore_index=True)

        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)
