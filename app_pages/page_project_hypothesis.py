import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"** Objective **\n"
        f"* Welcome to the Plant Disease Detection App! Our goal is to help farmers identify diseases in their crops early, promoting better crop management and increased yield.\n\n"
        f"**How it Works **\n"
        f"* Simply upload an image of a plant leaf, and our trained model will predict whether the plant is Healthy, has Powdery Mildew, or is affected by Rust.\n"
        f"* The trained model provides some suggestions for treating plant diseases.\n\n"
        f"** Dataset **\n"
        f"* We trained our model using a dataset of labeled images containing Healthy, Powdery, and Rust-afflicted plant leaves. This ensures accurate predictions for your images.\n\n"
        f"** Machine Learning Model **\n"
        f"* Our model, built with a convolutional neural network (CNN), excels at extracting features from plant images, providing precise disease classifications.\n\n"
    )

    st.info(
        f"** Success Criteria **\n"
        f"- **Accurate Predictions:** Our model achieves high accuracy in identifying plant diseases.\n"
        f"- **User-Friendly Interface:** The app is designed to be intuitive for farmers to use.\n"
        f"- **Early Disease Detection:** We aim to contribute to early detection of plant diseases.\n"
    )

    st.warning(
        f"** Potential Impact **\n"
        f"- **Improved Crop Yield:** Early disease detection can prevent the spread of diseases, positively impacting crop yield.\n"
        f"- **Informed Decision-Making:** Make data-driven decisions for effective crop management.\n"
    )

    st.success(
        f"** Thank you for using our Plant Disease Detection App! Your feedback is valuable to us. **\n"
        f'* Contact us via Email: amirhossein.bayani@gmail.com \n'
    )