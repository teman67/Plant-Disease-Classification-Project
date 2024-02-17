import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random

def page_plant_image_visualizer_body():
    '''
    Displays plant image visualizations including average and variability images,
    difference between average images, and image montage.
    '''

    st.write("### Plant image Visualizer")
    st.info(
        f"* The client is interested in having a study that visually "
        f"differentiates a healthy plant from a powdery or rust plant.")
    
    version = 'v1'
    if st.checkbox("Difference between average and variability image; Healthy vs. Powdery"):
      
      avg_healthy = plt.imread(f"jupyter_notebooks/outputs/{version}/avg_var_Healthy.png")
      avg_powdery = plt.imread(f"jupyter_notebooks/outputs/{version}/avg_var_Powdery.png")

      st.warning(
        f"* We notice the average and variability images did not show "
        f"patterns where we could intuitively differentiate one from another. " 
        f"However, a small difference in the colour pigment of the average images is seen for both labels.")

      st.image(avg_healthy, caption='Healthy Plant - Average and Variability')
      st.image(avg_powdery, caption='Powdery Plant - Average and Variability')
      st.write("---")

    if st.checkbox("Difference between average and variability image; Healthy vs. Rust"):
      
      avg_healthy = plt.imread(f"jupyter_notebooks/outputs/{version}/avg_var_Healthy.png")
      avg_rust = plt.imread(f"jupyter_notebooks/outputs/{version}/avg_var_Rust.png")

      st.warning(
        f"* We notice the average and variability images did not show "
        f"patterns where we could intuitively differentiate one from another. " 
        f"However, a small difference in the colour pigment of the average images is seen for both labels.")

      st.image(avg_healthy, caption='Healthy Plant - Average and Variability')
      st.image(avg_rust, caption='Rust Plant - Average and Variability')
      st.write("---")

    if st.checkbox("Differences between average healthy and average powdery plants"):
          diff_between_avgs = plt.imread(f"jupyter_notebooks/outputs/{version}/avg_diff_label1_label2.png")

          st.warning(
            f"* We notice this study didn't show "
            f"patterns where we could intuitively differentiate one from another.")
          st.image(diff_between_avgs, caption='Difference between average images')

    if st.checkbox("Differences between average healthy and average rust plants"):
          diff_between_avgs = plt.imread(f"jupyter_notebooks/outputs/{version}/avg_diff_label1_label3.png")

          st.warning(
            f"* We notice this study didn't show "
            f"patterns where we could intuitively differentiate one from another.")
          st.image(diff_between_avgs, caption='Difference between average images')

    if st.checkbox("Image Montage"): 
      st.write("* To refresh the montage, click on the 'Create Montage' button")
      my_data_dir = 'inputs/plants_dataset/Merged_split_images_swapped'
      labels = os.listdir(my_data_dir+ '/Test')
      label_to_display = st.selectbox(label="Select label", options=labels, index=0)
      if st.button("Create Montage"):      
        image_montage(dir_path= my_data_dir + '/Test',
                      label_to_display=label_to_display,
                      nrows=9, ncols=3, figsize=(10,25))
      st.write("---")



def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15,10)):
  '''
    Displays a montage of images from a specified directory and label.

    Args:
    - dir_path (str): The directory path where the images are located.
    - label_to_display (str): The label of the images to display.
    - nrows (int): Number of rows in the montage.
    - ncols (int): Number of columns in the montage.
    - figsize (tuple): Size of the figure for displaying the montage.
  '''
  
  sns.set_style("white")
  labels = os.listdir(dir_path)

  # subset the class you are interested to display
  if label_to_display in labels:

    # checks if your montage space is greater than subset size
    # how many images in that folder
    images_list = os.listdir(dir_path+'/'+ label_to_display)
    if nrows * ncols < len(images_list):
      img_idx = random.sample(images_list, nrows * ncols)
    else:
      print(
          f"Decrease nrows or ncols to create your montage. \n"
          f"There are {len(images_list)} in your subset. "
          f"You requested a montage with {nrows * ncols} spaces")
      return
    

    # create list of axes indices based on nrows and ncols
    list_rows= range(0,nrows)
    list_cols= range(0,ncols)
    plot_idx = list(itertools.product(list_rows,list_cols))


    # create a Figure and display images
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)
    for x in range(0,nrows*ncols):
      img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
      img_shape = img.shape
      axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
      axes[plot_idx[x][0], plot_idx[x][1]].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px")
      axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
      axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
    plt.tight_layout()
    
    st.pyplot(fig=fig)
    # plt.show()


  else:
    print("The label you selected doesn't exist.")
    print(f"The existing options are: {labels}")