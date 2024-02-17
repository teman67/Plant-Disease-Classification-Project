import streamlit as st
import matplotlib.pyplot as plt
import time

# Function to check if the warning message has been shown
def check_warning_message_state():
    '''
    Checks if the warning message has been shown.
    '''

    if 'warning_shown' not in st.session_state:
        st.session_state.warning_shown = False

# Function to show the warning message
def show_warning_message():
    '''
    Displays the warning message recommending light mode over dark mode.
    '''

    if not st.session_state.warning_shown:
        placeholder = st.empty()
        placeholder.markdown('<div style="background-color: #FFEEEB; padding: 30px; margin-top: 40px; border-radius: 5px; text-align: center;"><p style="font-size: 20px; color: #333333"><strong>For better visualization, it is recommended to use Light mode instead of Dark mode in Settings.</strong></p></div>', unsafe_allow_html=True)
        st.session_state.warning_shown = True

        time.sleep(5)  # Wait for 5 seconds
        placeholder.empty()


def page_summary_body():
    '''
    Displays the page summary body including project details and a warning message.
    '''

    check_warning_message_state()
    show_warning_message()

    st.write("### Quick Project Summary")

    st.info(
        f"**Introducing to the Plant Disease Recognition**\n"
        f"* The Plant Disease Classification System aims to assist farmers and agricultural professionals in identifying and managing plant diseases efficiently in apple trees.\n"
        f"* The system will focus on classifying apple plant images into three categories: Healthy, Rust Affected, and Powdery Mildew Affected.\n"
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
        f"The project has 3 business requirements:\n"
        f"* 1 - Accurately identify and classify plant diseases based on input images.\n"
        f"* 2 - Distinguishing between Healthy, Powdery, and Rust plants.\n"
        f"* 3 - Provide recommendations for treating plants based on the type of disease they are afflicted with."
        )

