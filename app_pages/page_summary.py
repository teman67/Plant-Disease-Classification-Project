import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**Introducing to the Plant Disease Recognition**\n"
        f"* The Plant Disease Classification System aims to assist farmers and agricultural professionals in identifying and managing plant diseases efficiently.\n"
        f"* The system will focus on classifying plant images into three categories: Healthy, Rust Affected, and Powdery Mildew Affected.\n"
        f"* Rusts are plant diseases caused by pathogenic fungi of the order Pucciniales (previously known as Uredinales).\n"
        f"Rusts get their name because they are most commonly observed as deposits of powdery rust-coloured or brown spores on plant surfaces, [Rust image](https://www.gardeningknowhow.com/wp-content/uploads/2020/11/plant-rust-disease.jpg).\n"
        f"* Powdery mildew is a fungal disease that affects a wide range of plants. Powdery mildew diseases are caused by many different species of fungi in the order Erysiphales."
        f"It is important to be aware of powdery mildew and its management as the resulting disease can significantly reduce important crop yields, [Powdery image](https://media.istockphoto.com/photos/grapevine-diseases-downy-mildew-is-a-fungal-disease-that-affects-a-picture-id1161364148?k=6&m=1161364148&s=612x612&w=0&h=BzE8nsZHyGD3y7r1wvKIYDrvqLQcJdk_efFCUNB3134=)\n"
        f"* Visual criteria are used to detect plant disease.\n"
        f"\n"
        f"**Project Dataset**\n"
        f"* The available dataset contains 1532 images divided into train, test, and validation sets.\n"
        f"* The datasets were taken from [Plant Disease Datasets](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset)")

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/teman67/PP5-Plant-Disease-Classification/blob/main/README.md).")
    

    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - Accurately identify and classify plant diseases based on input images.\n"
        f"* 2 - Distinguishing between Healthy, Powdery, and Rust plants. "
        )
