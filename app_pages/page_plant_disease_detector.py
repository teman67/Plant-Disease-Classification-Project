import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

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

    images_buffer = st.file_uploader('Upload blood smear samples. You may select more than one.',
                                     type='png', accept_multiple_files=True)

    # Define a dictionary to map class indices to class names
    class_mapping = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = (Image.open(image))
            st.info(f"Blood Smear Sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v1'
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
            
            df_report = df_report.append({"Name": image.name, 'Result': predicted_class_name},
                                         ignore_index=True)

        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)




