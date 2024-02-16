
# <h1 align="center">**Plant Disease Classification**</h1>

## Introduction

The Plant Disease Classification dashboard application utilizes Machine Learning technology to allow users to upload images of plant leaves for analysis. It assesses whether the plant is healthy or afflicted with powdery mildew or Rust, providing users with a downloadable report summarizing the findings.

![Home Screen](/readme/First_page.png)

### Deployed version at [Plant Disease Classification](https://plant-disease-classification-04c8092dc2fe.herokuapp.com/)

## Table of Contents

- [Business Requirements](#business-requirements)
- [Hypotheses and how to Validate](#hypotheses-and-how-to-validate)
- [Rational to Map Business Requirements](#the-rationale-to-map-the-business-requirements-to-the-data-visualisations-and-ml-tasks)
- [ML Business Case](#ml-business-case)
- [User Stories](#user-stories)
- [Dashboard Design](#dashboard-design---streamlit-app-user-interface)
- [Methodology](#methodology)
- [Rationale for the Model](#rationale-for-the-model)
- [Project Features](#project-features)
- [Project Outcomes](#project-outcomes)
- [Hypothesis Outcomes](#hypothesis-outcomes)
- [Testing](#testing)
- [Bugs](#bugs)
- [Deployment](#deployment)
- [Languages and Libraries](#languages-and-libraries)
- [Credits](#credits)

## Business Requirements

Farmy & Foods is facing a challenge with their apple leaf plantation, as both powdery mildew and rust, fungal diseases common in apple trees, have been detected. These diseases pose a threat to the quality and productivity of their apple leaf crops.

Currently, the inspection process involves manual assessment of each apple tree to identify signs of powdery mildew or rust. It takes time to collect leaf samples and visual them for symptoms of infection. Upon detection, specific compounds are applied to mitigate the spread of the diseases, taking additional time per tree. However, given the extensive apple leaf plantation spanning multiple farms, this manual process is not practical and lacks scalability.

To address this issue, the IT team has proposed implementing a Machine Learning system capable of swiftly detecting powdery mildew and rust using leaf images. This automated solution would streamline the inspection process, enabling rapid and accurate identification of infected trees. If successful, this initiative could be expanded to cover other crops, offering an efficient and scalable approach to managing pests and diseases across Farmy & Foods' agricultural operations.

Summary:

- The client is interested in conducting a study to visually differentiate a healthy apple leaf from one with powdery mildew and rust.
- The client is interested in a dashboard that predicts if a apple leaf is healthy or infeacted by powdery mildew or rust by 95% accuracy.
- The client would like to receive some treatments based on the type of plant diseases.

[Table Of Contents](#table-of-contents)

## Hypotheses and how to Validate

1. The identification of apple leaves affected by powdery mildew or rust from healthy leaves can be achieved through visual examination of their distinct appearances.
   - This can be confirmed through the creation of an average image study and image montage, allowing for a comparative analysis of the appearance differences between healthy leaves and those affected by powdery mildew or rust.
2. The determination of apple leaves as healthy or afflicted with powdery mildew or rust can be accomplished with a confidence level of 95% accuracy. 
   - This assertion can be substantiated by assessing the model's performance on the test dataset, aiming for a minimum accuracy rate of 95%.
3. The model's prediction accuracy may be compromised if the images of apple leaves contain backgrounds different from the beige background of the Kaggle dataset. 
   - o confirm this limitation, the model should be tested with new pictures of apple leaves featuring backgrounds distinct from those in the dataset images.
4. It is advisable to use images in RGB mode for improved prediction accuracy. Nevertheless, if images are not already in RGB mode, the trained model will automatically convert them to RGB mode for processing.

[Table Of Contents](#table-of-contents)

## The rationale to map the business requirements to the Data Visualisations and ML tasks

- Business Requirement 1: Data Visualization

  - The dashboard will showcase the 'mean' and 'standard deviation' images for both healthy and powdery mildew-infected apple leaves or for both healthy and rust infected apple leaves.
  - Additionally, it will display the contrast between an average healthy leaf and an average leaf infected with powdery mildew or rust.
  - Furthermore, an image montage featuring healthy leaves, leaves affected by powdery mildew, and rust leaves will be presented for comparison.

- Business Requirement 2: Classification

  - Create and fit a machine learning model to predict if a given leaf is healthy or infected with powdery mildew or rust. This will be a classification task with three classes and will require to set the image shape.
  - The model provides treatment recommendations to users based on the type of plant disease identified.
  - The predictions should have a 95% accuracy level.
  
- Business Requirement 3: Report
  - A downloadable report containing the predicted status of all uploaded images is available for users.

[Table Of Contents](#table-of-contents)

## ML Business Case

- To create a machine learning model for leaf classification, particularly distinguishing between healthy leaves and those infected with powdery mildew or rust, typically the following steps are followed:
1. Data Collection: Gather a dataset containing images of leaves categorized as healthy, powdery mildew-infected, and rust-infected. Ensure each category is well-represented in the dataset.

2. Data Preprocessing: Preprocess the images to ensure uniformity in size, color space, and quality. This step may involve resizing, normalization, and augmentation techniques to enhance the dataset's diversity.

3. Feature Extraction: Use techniques like convolutional neural networks (CNNs) to extract meaningful features from the images. CNNs are particularly effective for image classification tasks due to their ability to capture spatial hierarchies.

4. Model Selection: Choose an appropriate machine learning model architecture for classification. Common choices for image classification tasks include CNN-based architectures such as VGG, ResNet, or custom-designed models.

5. Model Training: Split the dataset into training and validation sets. Train the selected model on the training set while validating its performance on the validation set. Fine-tune hyperparameters to optimize the model's performance.

6. Model Evaluation: Evaluate the trained model's performance using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score on the validation set.

7. Model Testing: Once satisfied with the model's performance, test it on a separate test dataset to assess its generalization ability. This step ensures that the model can accurately classify unseen data.

8. Deployment: Deploy the trained model into production, making it available for inference on new leaf images. Integrate the model into an application or system where users can upload leaf images and receive predictions on their health status.

9. Monitoring and Maintenance: Continuously monitor the model's performance in production and update it periodically with new data to ensure its effectiveness over time.

- The dataset contains 1532 images taken from the client's crop fields. The images show healthy apple leaves, apple leaves that have powdery mildew or rust.
- The dataset is located on [Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset).

[Table Of Contents](#table-of-contents)

## User Stories

- As a client I require an intuitive dashboard for easy navigation, allowing me to effortlessly access and comprehend data, models, and outcomes.
- As a client I need the capability to observe average and variable images of both healthy apple leaves and those infected with powdery mildew and rust. This feature will enable me to visually distinguish between the three categories.
- As a client I seek the ability to view a visual montage comprising images of healthy apple leaves and those infected with powdery mildew or rust. This feature will facilitate a clearer differentiation between the three classifications.
- As a client I desire the functionality to upload images of apple leaves and receive classification predictions with an accuracy exceeding 95%. This will allow for swift assessment of apple tree health based on the provided predictions.
- As a client I require treatment suggestions based on identified plant diseases to effectively address any issues affecting my plants' health.
- As a client I require the facility to download a report containing the provided predictions, ensuring that I have a record of the outcomes for future reference.

[Table Of Contents](#table-of-contents)

## Dashboard Design - Streamlit App User Interface

### Page 1: Introduction

- Provide an overview of powdery mildew and rust, accompanied by sample images for illustration.
- Outline the specifics of the dataset utilized in the project.
- Define the business requirements.
- Include a hyperlink to access this Readme file.

### Page 2: Plant leaves Visualizer

- This page is designed to meet Business Requirement 1 by showcasing the following:
  - Illustrating the disparity between the average and variability image.
  - Displaying the contrast between average healthy leaves and leaves infected with powdery mildew or rust.
  - Presenting an image montage featuring for healthy leaves, leaves infected with powdery mildew, and leaves infected by rust.

### Page 3: Plant Diseases Detector

- This page is designed to meet Business Requirements 2 and 3, offering the following features:
  - Prediction of whether a leaf is infected with powdery mildew or rust.
  - Provision of a link to download a set of images displaying healthy leaves and leaves infected with powdery mildew or rust for live prediction.
  - User Interface featuring a file uploader widget for multiple leaf image uploads. It displays each uploaded image along with a prediction statement indicating if the leaf is infected with powdery mildew or rust, along with the associated probability.
  - In addition to uploading images directly from their device, users can also copy and paste image URL(s) from external sources for live prediction.
  - Generation of a report containing image names and prediction results.
  - Offering treatment recommendations tailored to each plant disease.
  - Download button provided to download the generated report.

### Page 4: Project Hypothesis and Validation

- Detail each [hypotheses](#hypotheses-and-how-to-validate), how it was validated and the conclusion.

### Page 5: Machine Learning Performance Metrics

- Providing comprehensive details on the model performance, including:
  - Label frequencies for the training, validation, and test sets.
  - Model history depicting accuracy and losses during training.
  - Evaluation results of the model's performance.
  - Offering metrics that demonstrate the performance of the model.

## Methodology

### CRISP-DM

CRISP-DM (Cross Industry Standard Process for Data Mining) methodology was used for the data mining project. There are six stages to the process and have the following relationship:

![CRISP_DM](readme_files/crisp-dm.png)

### Agile

An agile approach was implemented for the project using GitHub projects with the aid of Milestones and Issues. Each Issue detailed the relevant tasks to be completed.

The project board can be viewed [here](https://github.com/users/Porsil/projects/7)

[Table Of Contents](#table-of-contents)

## Rationale for the Model

A good model generates accurate predictions as the machine learning model generalizes well from the training data and allows for accurate predictions on unseen data. A good model is also as simple as possible and does not have an unnecessarily complex neural network or high computational power.

When a model trains for too long on the training dataset or if the model is too complex, it can start to learn the noise or irrelevant information from the dataset. This causes the model to fit too closely to the training set and become overfitted where it is unable to generalize well to new data. Overfitting can be detected by seeing how the model performs on the validation and test datasets.

Underfitting can also occur where the model cannot determine a meaningful relationship between the input and output data. Underfitting can be detected by seeing how the model performs on the training dataset, as the accuracy of the training dataset will be low. This is also be translated into low accuracy over the validation and test datasets.

### Model Creation

As this project is an image classification task, a Convolutional Neural Network (CNN) will be created using Tensorflow. The project requires a binary image classification model as the outcome can be one of two choices: healthy or infected.

There are two choices available for binary classification tasks. 1 neuron with sigmoid as activation function or 2 neurons with softmax as activation function. Both functions were used to create and fit models during the testing phase.

The model was created by trial and error, considering any underfitting or overfitting of previous versions to find a model that has a normal fit, refer to [testing](#testing). Based on the model evaluation data, version 6 was chosen for the dashboard.

The model created is a sequential model containing the following:

- Convolutional layers: used to select the dominant pixel value from the non-dominant pixels in images using filters to find patterns (or features) in the image.

  - 3 Convolution layers were used in the model.
  - Conv2D was chosen as the images are 2D.
  - The number of filters chosen was 32, 16 then 8 to keep the complexity low.
  - The kernel size used was 3x3 as this is regarded as the most efficient.
  - Activation 'Relu' used as it is simple and effective with hidden layers of a binary classification model.

- Pooling layers: used to reduce the image size by extracting only the dominant pixels (or features) from the image.
  - After each convolution layer is a pooling layer. The combination of these two layers removes the nonessential part of the image and reduces complexity.
  - MaxPooling was used as this selects the brighter pixels from the image i.e. the white (brighter pixel) powdery mildew on a green (darker pixels) leaf.
- Flatten layer: used to flatten the matrix into a vector, which means a single list of all values, that is fed into a dense layer.

- Dense layer: a fully-connected neural network layer.

  - 64 nodes were chosen through the trial and error process.
  - Activation 'Relu' used.

- Dropout layer: a regularization layer used to reduce the chance of overfitting the neural network.

  - 0.3 was used, which was deemed appropriate for the number of images available.

- Output layer:
  - Softmax was chosen as the activation function through the trial and error process. As such, 2 nodes were chosen as there are two output possibilities and categorical_crossentropy was chosen for loss.
  - Adam optimizer was chosen through the trial and error process.

![Model](readme_files/model.png)

[Table Of Contents](#table-of-contents)

## Project Features

<details>

<summary>Navigation</summary>

The navigation bar is visible on all dashboard pages and provides easy links to other pages.

![Menu](readme_files/menu.png)

</details>

<details>

<summary>Page 1: Introduction</summary>

The introduction page provides the user with information about powdery mildew, the project summary, the dataset and the business requirements. There is also a link to this ReadMe file.

![Introduction](readme_files/introduction.png)

</details>

<details>

<summary>Page 2: Leaf Visualizer</summary>

The leaf visualizer page provides the user with the results of the study to visually differentiate a healthy cherry leaf from one with powdery mildew. It was determined that healthy leaves and infected leaves could be distinguished by their appearance.

The page gives the user the options to view the difference between average and variability images, the differences between average infected and average uninfected leaves and an image montage of healthy or infected leaves.

![Visualizer_1](readme_files/visualizer_1.png)
![Visualizer_2](readme_files/visualizer_2.png)
![Visualizer_3](readme_files/visualizer_3.png)
![Visualizer_4](readme_files/visualizer_4.png)

</details>

<details>

<summary>Page 3: Powdery Mildew Detector</summary>

The detector page allows the user to upload images of cherry leaves to determine if the leaf is healthy or infected with powdery mildew. Each image is presented with a prediction and a graph depicting the probability of the predictions accuracy. There is then a report detailing the image name, probability accuracy and result. This report is available to download into a .csv file, which can be viewed easily in Microsoft Excel.

![Detector](readme_files/detector.png)

</details>

<details>

<summary>Page 4: Project Hypothesis</summary>

The hypothesis page provides the user with details of the project hypotheses and their outcomes.

![Hypothesis](readme_files/hypothesis.png)

</details>

<details>

<summary>Page 5: Performance Metrics</summary>

The performance metrics page provides the user with the Machine Learning model dataset distribution, performance plots and performance on the test dataset.

![Performance](readme_files/performance.png)

</details>

[Table Of Contents](#table-of-contents)

## Project Outcomes

### Business Requirement 1: Data Visualization

The visualization study can be viewed on the [Leaf Visualizer page](https://cherry-leaf-mildew-detection.herokuapp.com/) of the dashboard. The study shows the mean and variability images and an image montage of both healthy and infected leaves. The concludes that healthy leaves and infected leaves can be distinguished by their appearance as leaves infected with powdery mildew exhibit white marks.

### Business Requirement 2: Classification

The classification tool can found on the [Powdery Mildew Detector page](https://cherry-leaf-mildew-detection.herokuapp.com/) of the dashboard. The user is able to upload images of cherry leaves and is given a classification prediction for each image along with a probability graph. The predictions have an accuracy level of above 97%.

### Business Requirement 3: Report

The report can be viewed on the [Powdery Mildew Detector page](https://cherry-leaf-mildew-detection.herokuapp.com/) of the dashboard once images have been classified. The user is presented with a table that shows the image name, probability % and result for each uploaded image. The user can also click 'Download Report' which downloads the report to a .csv file, which can be opened easily in Microsoft Excel.

[Table Of Contents](#table-of-contents)

## Hypothesis Outcomes

### Hypothesis 1

- Cherry leaves with powdery mildew can de differentiated from healthy leaves by their appearance.

This hypothesis was validated by creating an average image study and image montage to determine differences in the appearance of healthy leaves and leaves affected with powdery mildew.

An image montage shows that leaves infected with powdery mildew are easily identified due to the present of white deposits on the infected leaves.
The average and variability images showed a pattern within the center of the leaf related to colour pigmentation. This is most notable in the variability images where the center of the healthy leaves looks black whereas the center for the infected leaves is not.

![Variability](readme_files/variability.png)

The difference between averages study did not show patterns where we could intuitively differentiate one from another.
The image montage, average and variability images and the difference between averages study can be viewed by selecting the 'Leaf Visualizer' option on the sidebar menu.

Conclusion: This hypothesis was correct and healthy leaves and infected leaves can be distinguished by their appearance as leaves infected with powdery mildew exhibit white marks.

### Hypothesis 2

- Cherry leaves can be determined to be healthy or contain powdery mildew with a degree of 97% accuracy.

This was validated by evaluating the model on the test dataset.

The model accuracy was trained at over 99% with the train and validation datasets, and 99% accuracy was achieved on the test dataset.

Conclusion: This hypothesis was correct as the model was successfully trained using a Convolutional Neural Network to classify if an image of a cherry leaf is healthy or infected with powdery mildew with a degree of accuracy of above 99%.

### Hypothesis 3

- If the image has a different background to the beige background of the Kaggle dataset the model will predict false results.

This was validated by uploading the following images to the dashboard:

![Hypothesis Pictures](readme_files/hypothesis3_images.png)

The results were 7 correct predictions and 3 incorrect predictions as follows:

![Hypothesis Results](readme_files/hypthesis3_results.png)

This insight will be taken to the client to ensure they are aware of the image background requirements for the best model performance.

Conclusion: This hypothesis was correct as the model incorrectly predicted the classification of 3 of the 10 images.

[Table Of Contents](#table-of-contents)

## Testing

<details>

<summary>ML Model Testing</summary>

The model testing can be viewed [here](readme_files/model_testing.pdf).

The version used for the dashboard was version 6, as this showed a normal fit with no sign of overfitting and had an accuracy level of above 97% to meet business requirement 2.

</details>

<details>

<summary>Dashboard Testing</summary>

| Page         |          Feature          | Pass / Fail |
| ------------ | :-----------------------: | :---------: |
| Introduction |          Content          |    Pass     |
| Introduction |         Nav link          |    Pass     |
| Introduction |        ReadMe link        |    Pass     |
| Visualizer   |          Content          |    Pass     |
| Visualizer   |    1st checkbox ticked    |    Pass     |
| Visualizer   |   1st checkbox unticked   |    Pass     |
| Visualizer   |    2nd checkbox ticked    |    Pass     |
| Visualizer   |   2nd checkbox unticked   |    Pass     |
| Visualizer   |    3rd checkbox ticked    |    Pass     |
| Visualizer   |   3rd checkbox unticked   |    Pass     |
| Visualizer   |      Healthy montage      |    Pass     |
| Visualiser   |     Infected montage      |    Pass     |
| Detector     |          Content          |    Pass     |
| Detector     |        Kaggle link        |    Pass     |
| Detector     | Drag and drop file upload |    Pass     |
| Detector     |    Browse file upload     |    Pass     |
| Detector     |   Show uploaded images    |    Pass     |
| Detector     |     Show predictions      |    Pass     |
| Detector     |  Show probability graph   |    Pass     |
| Detector     |      Analysis report      |    Pass     |
| Detector     |    Downloadable report    |    Pass     |
| Hypothesis   |          Content          |    Pass     |
| Performance  |          Content          |    Pass     |

</details>

[Table Of Contents](#table-of-contents)

## Bugs

### Fixed Bugs

<details>

<summary>CodeAnywhere</summary>

Several issues were encountered with the CodeAnywhere IDE.

Firstly, the IDE would often go offline for 2 to 3 seconds. For the most part this was not an issue but if this occurred whilst executing a cell in a Jupyter notebook the IDE would crash and require restarting. This meant it was particularly difficult to fit the model as successfully executing this process often took up to an hour. As such not as many models were trained as I would have hoped. I decided to stop after v5 and ensure my dashboard and ReadMe report were nearly complete to ensure my project would be submitted on time, before returning to create and fit further models. This issue also meant that early stopped was added to each model, as running all 25 epochs would take several hours and have a high risk of disconnection. This in turn meant that it was harder to detect if the model was overfitting. On return to model fitting, the import libraries was split into individual code cells as this seemed to cause the system to crash less often.

Secondly, during the model training impacted by the first issue, often the code that was saved and committed did not match the code what was in the workspace. Autosave was enabled and all code double checked to be saved by selecting File>Save All before the ‘git add’ and ‘git commit’ commands. This is shown in the commit for v4 where the code in Jupyter notebook 03_modelling_and_evaluation shows an error message for fitting the model, if this was true the outputs for the model would not have been generated.

Thirdly, when trying to commit the code for v3, the workspace would crash. Once re-opened the ‘git add’ command would produce the following error:

![IDE_error](readme_files/ide_error.png)

This was fixed by running the command ‘rm -f .git/index.lock’ as per this [Stack Overflow post](https://stackoverflow.com/questions/38004148/another-git-process-seems-to-be-running-in-this-repository). The issue kept occurring so instead of using ‘git add .’ each file was added individually. This determined that the source of the error was the ‘mildew_detector_model.h5’. To fix the error this file was added to the .gitignore file. The same was done for v2-5, and only v5 was removed to complete the dashboard.

</details>

<details>

<summary>Incorrect Softmax Predictions</summary>

The models using Softmax as the activation function initially gave the incorrect predictions as the outputs were reversed. This [Stack Overflow post](https://stackoverflow.com/questions/54377389/keras-imagedatagenerator-why-are-the-outputs-of-my-cnn-reversed) suggested the cause of the bug. To fix, the labels were explicitly added to the prediction code (below) and corrected in the src/machine_learning/predictive_analysis.py file.

![Softmax Bug](readme_files/softmax_bug.png)

</details>

### Unfixed Bugs

There are no known unfixed bugs.

[Table Of Contents](#table-of-contents)

## Deployment

### Heroku

To deploy this app to Heroku from its GitHub repository:

#### Create a Heroku App:

- Log in to [Heroku](https://dashboard.heroku.com/apps). If required, create an account.
- Click the 'New' button in the top right and select 'Create new app' from the drop-down menu.
- Enter a name for the app in the 'App name' field, this must be an unique and should be meaningful to the app's content.
- Select your region in the 'Choose a region' field.
- Click on the 'Create app' button.

#### Deploy in Heroku

- Ensure requirements.txt file exists and contains the dependencies.
- Set the stack to Heroku-20 as follows:
  - In Heroku, click 'Account Settings' from the avatar menu.
  - Scroll to the 'API Key' section and click 'Reveal' then copy the key.
  - In the workspace, Log in to the Heroku command line interface using 'heroku login -i'.
  - Enter your email and copied API when prompted.
  - Use the command 'heroku stack:set heroku-20 -a yourappname'. yourappname is the name given to the app in the 'Create a Heroku App' section above.
- Ensure the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- Ensure a Procfile is present and contains the code 'web: sh setup.sh && streamlit run app.py'.
- Ensure the code is committed and pushed to GitHub.
- In Heroku click on the 'Deploy' tab and scroll down to the 'Deployment Method' section. Select 'GitHub' and confirm you wish to deploy using GitHub. Enter your GitHub password if prompted.
- Scroll to the 'Connect to GitHub' section and search for your repository.
- Click 'Connect' when found.
- To deploy go to the 'Manual Deploy' section add the 'main' branch to 'Choose a branch to deploy' field and click 'Deploy Branch'.
- The app is now live, click 'View' to view the deployed site.

### Forking the repository

- Open the [Mildew Detection in Cherry Leaves](https://github.com/Porsil/mildew_detection_in_cherry_leaves) repository.
- Click the 'Fork' button in the top right.
- This creates a copy of the repository.

### Cloning the repository

- Open the [Mildew Detection in Cherry Leaves](https://github.com/Porsil/mildew_detection_in_cherry_leaves) repository.
- Click the green '<> Code' button. Select the preferred cloning option from the list then copy the link provided.
- Change the current working directory to the location where you want the cloned directory.
- Type 'git clone' and paste the URL you copied earlier.
- Press 'Enter' to create your local clone.

[Table Of Contents](#table-of-contents)

## Languages and Libraries

### Languages Used

- Python

### Frameworks, Libraries & Programs Used

- [GitHub](https://github.com/) was used for version control and agile methodology.
- [CodeAnywhere](https://codeanywhere.com/) was the workspace used for the project.
- [Kaggle](https://www.kaggle.com/) was the source of the dataset.
- [Jupyter Notebook](https://jupyter.org/) was used to run the machine learning pipeline.
- [Joblib](https://joblib.readthedocs.io/en/latest/) for saving and loading image shape.
- [NumPy](https://numpy.org/) was used to convert images into an array.
- [Pandas](https://pandas.pydata.org/) was used for data analysis and manipulation.
- [Matplotlib](https://matplotlib.org/) was used to create charts and plots.
- [Seaborn](https://seaborn.pydata.org/) was used for data visualization.
- [Plotly](https://plotly.com/) was used to create charts and plots.
- [Streamlit](https://streamlit.io/) was used to create the dashboard.
- [Scikit-learn](https://scikit-learn.org/stable/) was used as a machine learning library.
- [Tensorflow](https://www.tensorflow.org/) was used as a machine learning library.
- [Keras](https://keras.io/) was used as a machine learning library.
- [Heroku](https://dashboard.heroku.com/login) was used to deploy the site.

[Table Of Contents](#table-of-contents)

## Credits

- Code Institute [Malaria Detector](https://github.com/Code-Institute-Solutions/WalkthroughProject01) project was used extensively as a reference when creating this project.
- Code Institute [Mildew Detection in Cherry Leaves](https://github.com/Code-Institute-Solutions/milestone-project-mildew-detection-in-cherry-leaves) template was used to create the project.
- Code Institue lessons on Data Analytics Packages > ML:TensorFlow
- This [StackOverflow](https://stackoverflow.com/questions/38004148/another-git-process-seems-to-be-running-in-this-repository) post was used to fix the git add bug.
- This [StackOverflow](https://stackoverflow.com/questions/54377389/keras-imagedatagenerator-why-are-the-outputs-of-my-cnn-reversed) post was used to fix the softmax bug.
- Details of powdery mildew were taken from this [Wikipedia](https://en.wikipedia.org/wiki/Powdery_mildew) article.
- CRISP-DM diagram taken from [Data Science Process Alliance](https://www.datascience-pm.com/crisp-dm-2/).

[Table Of Contents](#table-of-contents)


