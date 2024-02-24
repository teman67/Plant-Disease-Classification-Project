
# <h1 align="center">**Plant Disease Classification**</h1>

## Introduction

The Plant Disease Classification dashboard application utilizes Machine Learning technology to allow users to upload images of apple leaves for analysis. It assesses whether the apple leaves is healthy or afflicted with powdery mildew or Rust, providing users with a downloadable report summarizing the findings.

![Home Screen](/readme/First_page_new.png)

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

## Hypotheses and how to Validate

1. The identification of apple leaves affected by powdery mildew or rust from healthy leaves can be achieved through visual examination of their distinct appearances.
   - This can be confirmed through the creation of an average image study and image montage, allowing for a comparative analysis of the appearance differences between healthy leaves and those affected by powdery mildew or rust.
2. The determination of apple leaves as healthy or afflicted with powdery mildew or rust can be accomplished with a confidence level of 95% accuracy. 
   - This assertion can be substantiated by assessing the model's performance on the test dataset, aiming for a minimum accuracy rate of 95%.
3. The model's prediction accuracy may be compromised if the images of apple leaves contain backgrounds different from the beige background of the Kaggle dataset. 
   - o confirm this limitation, the model should be tested with new pictures of apple leaves featuring backgrounds distinct from those in the dataset images.
4. It is advisable to use images in RGB mode for improved prediction accuracy. Nevertheless, if images are not already in RGB mode, the trained model will automatically convert them to RGB mode for processing.


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


## User Stories

- As a client I require an intuitive dashboard for easy navigation, allowing me to effortlessly access and comprehend data, models, and outcomes.
- As a client I need the capability to observe average and variable images of both healthy apple leaves and those infected with powdery mildew and rust. This feature will enable me to visually distinguish between the three categories.
- As a client I seek the ability to view a visual montage comprising images of healthy apple leaves and those infected with powdery mildew or rust. This feature will facilitate a clearer differentiation between the three classifications.
- As a client I desire the functionality to upload images of apple leaves and receive classification predictions with an accuracy exceeding 95%. This will allow for swift assessment of apple tree health based on the provided predictions.
- As a client I require treatment suggestions based on identified plant diseases to effectively address any issues affecting my plants' health.
- As a client I require the facility to download a report containing the provided predictions, ensuring that I have a record of the outcomes for future reference.


## Dashboard Design - Streamlit App User Interface

### Page 1: Quick Project Summary

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

CRISP-DM (Cross Industry Standard Process for Data Mining) methodology was employed for the data mining project, consisting of six stages with the following interrelated relationship:

![CRISP_DM](readme/crisp-dm.png)

### Agile

For the project, an agile approach was adopted, facilitated by GitHub projects with the assistance of Milestones and Issues. Each Issue comprehensively outlined the relevant tasks to be completed.

The project board can be viewed [here](https://github.com/teman67/PP5-Plant-Disease-Classification/issues)

### Image preparation

The original images [Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset) possess a large size and resolution of (2700, 3986, 3). To address this, we utilize the PIL package to resize and adjust the images, reducing their dimensions to (173, 256, 3). It's crucial to maintain a fixed aspect ratio for each image during this process.

## Rationale for the Model
- The rationale behind selecting a specific model for a machine learning task is typically based on several factors:

  - A good model excels in generating accurate predictions by effectively generalizing from the training data, thus enabling precise predictions on unseen data. Additionally, it maintains simplicity, avoiding unnecessary complexity in neural network architecture or computational power.

  - However, when a model undergoes excessive training or becomes overly complex, it risks learning noise or irrelevant information from the dataset. This phenomenon, known as overfitting, results in the model fitting too closely to the training set, leading to poor generalization on new data. Overfitting can be identified by assessing the model's performance on validation and test datasets.

  - Conversely, underfitting occurs when the model fails to discern meaningful relationships between input and output data. Detection of underfitting involves evaluating the model's performance on the training dataset, typically indicated by low accuracy. This deficiency also translates to low accuracy across validation and test datasets.

### Model Creation

- For this image classification project, a Convolutional Neural Network (CNN) is implemented using TensorFlow. The task involves classifying images into one of three categories: healthy, powdery mildew-infected, or rust-infected. Here's a breakdown of the model:

  - The model is initiated using the Sequential() function, indicating a sequential layer-by-layer architecture.

  - Four convolutional layers (Conv2D) are added successively to extract features from input images. Each convolutional layer has a 3x3 filter size and employs the Rectified Linear Unit (ReLU) activation function to introduce non-linearity. The first layer specifies an input shape of (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).

  - After each convolutional layer, a max-pooling layer (MaxPooling2D) with a 2x2 window size is added to downsample the feature maps, retaining the most salient features.

  - Following the convolutional and max-pooling layers, a Flatten() layer is included to flatten the feature maps into a one-dimensional vector, preparing them for input to the dense layers.

  - Two fully connected (Dense) layers are added with 128 units each, employing ReLU activation functions. These layers serve as intermediate layers for feature transformation and extraction.

  - The final dense layer consists of 3 units, representing the number of classes (Healthy, Powdery, Rust) in the classification task. It uses the softmax activation function to output probabilities for each class, ensuring the sum of probabilities across all classes equals 1.

The model architecture was iteratively refined through trial and error, aiming to address issues such as underfitting or overfitting observed in previous versions. The chosen model version, referred to as version 2 in the evaluation phase, demonstrated a balanced fit.

Based on the evaluation results, [version 2](/jupyter_notebooks/outputs/v2/) was selected for integration into the dashboard. Detailed insights into the testing phase can be found in the [testing section](#testing).

![ML_Model](/readme/ML_model.png)

## Project Features

<details>

<summary>Navigation</summary>

The navigation bar remains visible across all pages of the dashboard, offering convenient access to various sections.

![Menu](/readme/menu.png)

</details>

<details>

<summary>Page 1: Quick Project Summary</summary>

The Quick Project Summary page furnishes users with details regarding powdery mildew, rust, a project summary, dataset information, and the business requirements. Additionally, a hyperlink to access this ReadMe file is provided.

![Page_1](/readme/Page_1.png)

</details>

<details>

<summary>Page 2: Plant image Visualizer</summary>

The leaf visualizer page presents users with the outcomes of the study, aiding in visually distinguishing between a healthy apple leaf and one affected by powdery mildew or rust. It was established that healthy and infected leaves exhibit discernible differences in appearance.

On this page, users have the options to:

  - View the disparity between average and variability images.
  - Compare the distinctions between average infected and average uninfected leaves.
  - Access an image montage showcasing healthy or infected leaves.

![Page_2](/readme/Page_2.png)

</details>

<details>

<summary>Page 3: Plant Disease Detector</summary>

On the detector page, users have the capability to upload images of apple leaves to determine their health status, whether they are healthy or infected with powdery mildew or rust. Following the upload, each image is accompanied by a prediction statement and a graphical representation depicting the probability of the prediction's accuracy. Moreover, treatment suggestions tailored to the respective plant diseases are provided for user reference. Finally, a downloadable report is available, offering comprehensive details including the image name, probability accuracy, result, and corresponding treatment suggestions in a .csv format.

![Page_3](/readme/Page_3.png)

</details>

<details>

<summary>Page 4: Project Hypothesis and Validation</summary>

The hypothesis page furnishes users with comprehensive details regarding the project hypotheses and their respective outcomes.

![Page_4](/readme/Page_4.png)

</details>

<details>

<summary>Page 5: Machine Learning Performance</summary>

The performance metrics page offers users insights into various aspects of the machine learning model's performance, including:

1. Dataset distribution: Visual representation of the distribution of the dataset used for training, validation, and testing.

2. Performance plots: Graphical representations illustrating the performance metrics such as accuracy, loss, precision, recall, and F1-score during model training and evaluation.

3. Confusion matrix: A tabular representation that displays the model's predictions versus the actual labels across different classes, providing a comprehensive overview of the model's performance in terms of true positives, true negatives, false positives, and false negatives.

4. Performance on the test dataset: Summary of the model's performance metrics specifically on the test dataset, providing users with an understanding of how well the model generalizes to unseen data.

![Page_5](/readme/Page_5.png)

</details>

## Project Outcomes

### Business Requirement 1: Data Visualization

The visualization study can be accessed via the [Plant image Visualizer page](https://plant-disease-classification-04c8092dc2fe.herokuapp.com/) on the dashboard. This study showcases the mean and variability images alongside an image montage featuring both healthy and infected leaves. The study concludes that distinguishing between healthy and infected leaves is possible based on appearance.

### Business Requirement 2: Classification

- The classification tool is accessible on the [Plant Disease Detector page](https://plant-disease-classification-04c8092dc2fe.herokuapp.com/) of the dashboard. Users can upload images of apple leaves and receive a classification prediction for each image, accompanied by a probability graph. Notably, the predictions boast an accuracy level exceeding 95%.
- The prediction is accompanied by suggestions for plant diseases.

### Business Requirement 3: Report

The report is accessible on the [Plant Disease Detector page](https://plant-disease-classification-04c8092dc2fe.herokuapp.com/) of the dashboard following the classification of images. Users are presented with a table displaying the image name, probability percentage, corresponding treatment suggestions and result for each uploaded image. Additionally, users can download the report by clicking 'Download Report', saving it as a .csv file.

## Hypothesis Outcomes

### Hypothesis 1

- The hypothesis that apple leaves with powdery mildew or rust can be differentiated from healthy leaves by their appearance was confirmed through various visual analyses.

![Sample](/readme/sample.png)

An average image study and image montage were conducted to discern disparities between healthy leaves and those affected by powdery mildew or rust. The image montage vividly illustrated that leaves infected with powdery mildew are distinguishable due to the presence of white deposits on the affected leaves. Affected leaves with rust are distinguishable by the presence of yellow or orange spots on the plants.

Furthermore, the average and variability images revealed distinctive patterns, particularly within the center of the leaf, relating to color pigmentation. Notably, the variability images showcased a discernible difference wherein the center of healthy leaves appeared black, whereas the center of infected leaves did not exhibit the same characteristic.

![Variability](/readme/Variability.png)

The difference between averages study did not reveal discernible patterns that would facilitate intuitive differentiation between healthy and infected leaves. However, the image montage, average and variability images, and the difference between averages study can be accessed by selecting the 'Plant image Visualizer' option on the sidebar menu.

In conclusion, the hypothesis was validated, confirming that healthy leaves and infected leaves can indeed be distinguished by their appearance. Leaves affected with powdery mildew exhibit distinctive white marks, while rust manifests as yellow or orange spots.

### Hypothesis 2

- Apple leaves can be accurately classified as healthy or infected with powdery mildew or rust with a remarkable degree of 95% accuracy.

This assertion was substantiated by evaluating the model's performance on the test dataset.

The model exhibited outstanding accuracy during training, surpassing 95% with both the train and validation datasets. Furthermore, it achieved a remarkable 95% accuracy on the test dataset.

In conclusion, this hypothesis was verified as the model, trained using a Convolutional Neural Network, successfully classified images of cherry leaves as healthy or infected with powdery mildew or rust with an accuracy exceeding 95%.

### Hypothesis 3

- If the image contains a background that differs from the beige background of the Kaggle dataset, the model may produce inaccurate predictions.

This assertion was confirmed through the uploading of the following images to the dashboard:

![Hypothesis Pictures](/readme/test_plants.png)

The results were 5 correct predictions and 5 incorrect predictions as follows:

| Image Numbers | Classification |
|---------------|----------------|
| 1             | Healthy        |
| 2             | Healthy        |
| 3             | Healthy        |
| 4             | Rust           |
| 5             | Rust           |
| 6             | Rust           |
| 7             | Rust           |
| 8             | Healthy        |
| 9             | Healthy        |
| 10            | Rust           |

This insight will be conveyed to the client to ensure they understand the significance of adhering to the image background requirements for optimal model performance.

In conclusion, this hypothesis was validated as the model inaccurately predicted the classification of 5 out of 10 images.

## Testing

<details>

<summary>Low Accuracy of ML model</summary>

The first ML model [version 1](/jupyter_notebooks/outputs/v1/) had low accuracy and high loss as shown below:

![Accuracy](/jupyter_notebooks/outputs/v1/model_training_acc_old.png)

![Loss](/jupyter_notebooks/outputs/v1/model_training_losses_old.png)

I discovered that the reason for the issue was due to using: `model.add(Dense(1, activation='sigmoid'))`. To enhance my accuracy, I replaced it with `model.add(layers.Dense(3, activation='softmax'))` in [version 2](/jupyter_notebooks/outputs/v2/).

Using a sigmoid activation function in the output layer along with a single output neuron is typically suitable for binary classification tasks, where the goal is to predict between two classes (e.g., healthy vs. diseased). However, when dealing with multiple classes (e.g., healthy, powdery mildew, rust), it's recommended to use the softmax activation function in the output layer and have one output neuron per class.

Therefore, for a classification problem with three classes, it's preferable to use the softmax activation function in the output layer and have three output neurons, each representing one of the classes. This allows the model to output a probability distribution over the three classes, making it more appropriate for multi-class classification tasks.

</details>

<details>

<summary>User Story Testing</summary>

**Business Requirement 1: Data Visualization**

1. **Interactive Dashboard Navigation:**
   - **User Story:** Client needs easy navigation within the interactive dashboard for effective data comprehension.
   - **Test Scenario:**
     | Feature           | Action                           | Expected Outcome                         | Actual Outcome |
     | ----------------- | -------------------------------- | ---------------------------------------- | -------------- |
     | Navigation bar    | Clicking on side menu buttons    | Correct display of selected page content | Functions as intended |

2. **Plant Image Visualizer Page:**
   - **User Story:** Clients require visual representation of average images, image disparities, and variations between healthy and diseased apple leaves for quick identification.
   - **Test Scenarios:**
     | Feature                                | Action                            | Expected Outcome                      | Actual Outcome |
     | -------------------------------------- | --------------------------------- | ------------------------------------- | -------------- |
     | Checkbox for average and variability images | Checking the checkbox          | Rendering of relevant image plots     | Functions as intended |
     | Checkbox for difference between average images | Checking the checkbox       | Rendering of relevant image plots     | Functions as intended |
     | Checkbox for image montage            | Checking the checkbox           | Display of label selection dropdown and "Create montage" button | Functions as intended |
     | Image montage creation button         | Clicking 'Create Montage' button after label selection | Display of relevant image montage with correct label | Functions as intended |

**Business Requirement 2 and 3: Classification and Providing recommendations**

1. **Plant Disease Detector Page:**
   - **User Story:** Clients aim to upload apple leaf images to utilize the ML model for immediate and accurate disease prediction.
   - **Test Scenario:**
     | Feature            | Action                                          | Expected Outcome                            | Actual Outcome |
     | ------------------ | ----------------------------------------------- | ------------------------------------------- | -------------- |
     | File uploader      | Uploading image data via 'Browse files' button or using image URL(s) | Display of disease prediction with probabilities and suggest treatments | Functions as intended |

2. **Saving Model Predictions:**
   - **User Story:** Clients need to save model predictions in a CSV file with timestamps for record-keeping.
   - **Test Scenario:**
     | Feature                  | Action                                          | Expected Outcome                            | Actual Outcome |
     | ------------------------ | ----------------------------------------------- | ------------------------------------------- | -------------- |
     | Download Report link     | Clicking on 'Download Analysis Report as CSV' link | Saving of a CSV file with timestamps and prediction details | Functions as intended |

</details>

<details>

<summary>Dashboard Testing</summary>

| Page         |          Feature          | Pass / Fail |
| ------------ | :-----------------------: | :---------: |
| Quick Project Summary |          Content          |    Pass     |
| Quick Project Summary |         Nav link          |    Pass     |
| Quick Project Summary |        ReadMe link        |    Pass     |
| Plant Visualizer   |          Content          |    Pass     |
| Plant Visualizer   |    1st checkbox ticked    |    Pass     |
| Plant Visualizer   |   1st checkbox unticked   |    Pass     |
| Plant Visualizer   |    2nd checkbox ticked    |    Pass     |
| Plant Visualizer   |   2nd checkbox unticked   |    Pass     |
| Plant Visualizer   |    3rd checkbox ticked    |    Pass     |
| Plant Visualizer   |   3rd checkbox unticked   |    Pass     |
| Plant Visualizer   |    4th checkbox ticked    |    Pass     |
| Plant Visualizer   |   4th checkbox unticked   |    Pass     |
| Plant Visualizer   |    5th checkbox ticked    |    Pass     |
| Plant Visualizer   |   5th checkbox unticked   |    Pass     |
| Plant Visualizer   |      Healthy montage      |    Pass     |
| Plant Visualiser   |  Powdery Infected montage |    Pass     |
| Plant Visualizer   |   Rust Infected montage   |    Pass     |
| Plant Detector     |          Content          |    Pass     |
| Plant Detector     |        Kaggle link        |    Pass     |
| Plant Detector     |       Dropdown menu       |    Pass     |
| Plant Detector     |    Browse file upload     |    Pass     |
| Plant Detector     |    Copy & Paste URL(s)    |    Pass     |
| Plant Detector     |   Show uploaded images    |    Pass     |
| Plant Detector     |     Show predictions      |    Pass     |
| Plant Detector     |  Show probability graph   |    Pass     |
| Plant Detector     |     Suggest treatment     |    Pass     |
| Plant Detector     |      Analysis report      |    Pass     |
| Plant Detector     |    Downloadable report    |    Pass     |
| Project Hypothesis   |          Content          |    Pass     |
| Machine Learning Performance  |          Content          |    Pass     |

</details>

## Bugs

### Fixed Bugs

<details>

<summary>Gitpod & Github</summary>

Firstly, the default python version on gitpod is 3.10.12 and most of the packages in requirements.txt file could not be installed using python version 3.10.12. To solve the isuue, I first upgraded the python version using: 'pyenv install 3.8.18'. Then to change the default python version from 3.10.12 to 3.8.18, I use: 'pyenv global 3.8.18'. Then I run pip3 install -r requirements.txt to install all packages. 

Secondly, the size of the ML model version 1 was more than 100MB and I could not push it to github since the maximum allowed file size is 100MB. I use Git LFS as suggested in https://git-lfs.com/ to solve the problem.

</details>

<details>

<summary>Name of Uploaded Plant Image</summary>

There was an issue in the code where uploading images via URL caused a discrepancy in the Plant Image names. This occurred because the names of images uploaded from browsing files were replaced with those from the URL, resulting in a shift in names. To resolve this, I implemented a method to track the number of images already uploaded and adjusted the index accordingly when appending filenames extracted from URLs.

</details>

<details>

<summary>Plant Image Sizes</summary>

The original dataset from [kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset) has high-resolution images, which negatively impacts the speed performance of the machine learning model. Additionally, the size of the saved ML model was excessively large. To mitigate these issues, I utilized [Pillow](https://pypi.org/project/pillow/) for image manipulation.

</details>

<details>

<summary>RGB mode</summary>

The Streamlit app encountered an issue where it crashed upon uploading an image with a mode that was not RGB. To address this, I implemented a check to first determine the mode of the uploaded images. If the mode was not in RGB, the code automatically converted them to RGB mode.

</details>

### Unfixed Bugs

There are no known unfixed bugs.

## Deployment

### Heroku

To deploy this app to Heroku from its GitHub repository, follow these steps:

#### Create a Heroku App:

- Begin by logging in to [Heroku](https://dashboard.heroku.com/apps). If you don't have an account yet, you'll need to create one.
- Once logged in, click the 'New' button at the top right corner and select 'Create new app' from the drop-down menu.
- Choose a unique and descriptive name for your app in the 'App name' field.
- Select the region closest to your location from the 'Choose a region' field.
- After filling in the details, click on the 'Create app' button.

#### Deploying on Heroku

- Make sure your project includes a requirements.txt file listing all the dependencies required to run the app.
- Set the stack to Heroku-20 by following these steps:
  - Navigate to 'Account Settings' from the avatar menu in Heroku.
  - Under the 'API Key' section, click 'Reveal' to display your API key, then copy it.
  - In your command line interface, log in to Heroku using 'heroku login -i', and enter your email and copied API key when prompted.
  - Execute the command 'heroku stack:set heroku-20 -a yourappname', replacing 'yourappname' with the name you gave to your app during its creation.
- Ensure the runtime.txt file specifies a Python version supported by the [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack.
- Confirm the presence of a Procfile containing the command 'web: sh setup.sh && streamlit run app.py'.
- Ensure all your code changes are committed and pushed to GitHub.
- Back in Heroku, navigate to the 'Deploy' tab, and under the 'Deployment Method' section, select 'GitHub' and confirm your intention to deploy from GitHub. If prompted, enter your GitHub password.
- Search for your repository under the 'Connect to GitHub' section, and once found, click 'Connect'.
- To initiate deployment, proceed to the 'Manual Deploy' section, add the 'main' branch to the 'Choose a branch to deploy' field, and click 'Deploy Branch'.
- Your app is now live on Heroku. Click 'View' to access the deployed site.

### Forking the Repository

To create a copy of the repository, follow these steps:

- Visit the [Plant Disease Classification](https://github.com/teman67/PP5-Plant-Disease-Classification) repository.
- Click the 'Fork' button located at the top right corner of the page.

### Cloning the Repository

To clone the repository to your local machine, proceed as follows:

- Visit the [Plant Disease Classification](https://github.com/teman67/PP5-Plant-Disease-Classification) repository.
- Click the green '<> Code' button and select your preferred cloning option from the list. Copy the provided link.
- Open your terminal and navigate to the directory where you want to store the cloned repository.
- Execute the command 'git clone' followed by the URL you copied earlier.
- Press 'Enter' to create a local clone of the repository.

## Languages and Libraries

### Languages Used

- Python

### Frameworks, Libraries & Programs Used

- [GitHub](https://github.com/): GitHub served as our primary platform for version control, allowing us to collaboratively manage code changes and adopt agile methodology practices. It facilitated seamless collaboration among team members by providing features such as issue tracking, pull requests, and project boards.

- [Gitpod](https://gitpod.io/): Gitpod was the chosen workspace environment for our project. It offered a cloud-based integrated development environment (IDE) that allowed developers to easily spin up pre-configured development environments directly from GitHub repositories. This enabled team members to have consistent development setups and reduced the overhead of setting up individual development environments.

- [Kaggle](https://www.kaggle.com/): Kaggle served as the primary source of our dataset. Kaggle is a popular platform for data science and machine learning competitions, as well as a repository of datasets. Leveraging Kaggle allowed us to access high-quality datasets, explore community-contributed datasets, and participate in competitions to benchmark our models against others in the field.

- [Jupyter Notebook](https://jupyter.org/): Jupyter Notebook was utilized to execute our machine learning pipeline. Jupyter Notebook is an open-source web application that allows users to create and share documents containing live code, equations, visualizations, and narrative text. It provided an interactive computing environment that facilitated iterative development and experimentation with our machine learning models.

- [Joblib](https://joblib.readthedocs.io/en/latest/): Joblib was employed for saving and loading image shapes within our project. Joblib is a library in Python that provides utilities for saving and loading Python objects to/from disk efficiently. In our context, it helped streamline the process of persisting image shapes, allowing us to cache results and avoid redundant computations during our workflow.

- [NumPy](https://numpy.org/): NumPy was used for converting images into arrays. NumPy is a fundamental package for scientific computing in Python, providing support for multidimensional arrays, matrices, and high-level mathematical functions. It played a crucial role in preprocessing our image data, converting raw image data into numerical arrays that could be fed into our machine learning models.

- [Pandas](https://pandas.pydata.org/): Pandas was instrumental for data analysis and manipulation tasks within our project. Pandas is a powerful data manipulation and analysis library for Python, offering data structures and operations for manipulating numerical tables and time series data. It facilitated tasks such as data cleaning, transformation, and exploration, enabling us to gain insights from our dataset effectively.

- [Matplotlib](https://matplotlib.org/): Matplotlib was utilized for creating charts and plots to visualize our data. Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It provided us with a wide range of plotting functionalities, allowing us to generate various types of charts and plots to explore patterns and trends within our dataset.

- [Seaborn](https://seaborn.pydata.org/): Seaborn was employed for data visualization, complementing Matplotlib with additional high-level interfaces for drawing attractive and informative statistical graphics. Seaborn builds on top of Matplotlib and integrates seamlessly with Pandas data structures, making it easy to generate complex visualizations with concise code syntax.

- [Plotly](https://plotly.com/): Plotly was another tool utilized for creating interactive charts and plots in our project. Plotly is a versatile graphing library that supports a wide range of chart types and provides interactivity features such as hover tooltips, zooming, and panning. Its integration with Jupyter Notebook allowed us to create interactive visualizations directly within our notebooks.

- [Streamlit](https://streamlit.io/): Streamlit was used to create the dashboard for our project. Streamlit is an open-source app framework for building and sharing data-driven web applications quickly. It provided a simple and intuitive way to convert Python scripts into interactive web apps, enabling us to deploy our machine learning models and visualizations with minimal effort.

- [Scikit-learn](https://scikit-learn.org/stable/): Scikit-learn served as a cornerstone machine learning library within our project. Scikit-learn is a powerful and easy-to-use library for machine learning in Python, offering tools for data preprocessing, feature extraction, model selection, and evaluation. It provided a consistent interface for implementing various machine learning algorithms and pipelines, facilitating the development and evaluation of predictive models.

- [Tensorflow](https://www.tensorflow.org/): Tensorflow was another key machine learning library utilized in our project. Tensorflow is an open-source deep learning framework developed by Google, offering comprehensive support for building and deploying machine learning models at scale. It provided low-level APIs for neural network development as well as high-level APIs through its Keras interface, enabling us to build and train complex neural network architectures efficiently.

- [Pillow](https://pypi.org/project/pillow/): Pillow was employed for image manipulation tasks within our project. Pillow is a fork of the Python Imaging Library (PIL), providing support for opening, manipulating, and saving many different image file formats. It offered a rich set of functionalities for image processing, such as resizing, cropping, filtering, and enhancing, allowing us to preprocess images effectively before feeding them into our machine learning pipeline.

- [Keras](https://keras.io/): Keras was used as a high-level API for building neural networks within our project. Keras is a user-friendly interface for neural network development, providing a simple and intuitive way to design and train deep learning models. It abstracts away the complexities of low-level neural network implementation, allowing us to focus on model architecture and experimentation.

- [Heroku](https://dashboard.heroku.com/login): Heroku was utilized to deploy the web application for our project. Heroku is a cloud platform that enables developers to build, deploy, and scale applications quickly and efficiently. It provided a seamless deployment process, allowing us to deploy our Streamlit dashboard and associated machine learning models with minimal configuration and management overhead.

## Credits

- Code Institute [Malaria Detector](https://github.com/Code-Institute-Solutions/WalkthroughProject01) project was used extensively as a reference when creating this project.
- The readme template was taken from [Mildew Detection in Cherry Leaves](https://github.com/Porsil/mildew_detection_in_cherry_leaves/blob/main/README.md). 
- Dataset was taken from [Plant disease recognition dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset) under [CC0: Public Domain License](https://creativecommons.org/publicdomain/zero/1.0/). It means the person who associated a work with this deed has dedicated the work to the public domain by waiving all of his or her rights to the work worldwide under copyright law, including all related and neighboring rights, to the extent allowed by law. Anyone can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.
- CRISP-DM diagram taken from [Medium](https://medium.com/@yennhi95zz/6-the-deployment-phase-of-the-crisp-dm-methodology-a-detailed-discussion-f802a7cb9a0f).
- Youtube channel [Fanilo Andrianasolo](https://www.youtube.com/@andfanilo) was used for customization of the Streamlit app.
- Youtube video [Python | Resize Image with Pillow](https://www.youtube.com/watch?v=p9Wmk7YC_vs) was used to resize the downloaded images from [Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset) as explained in [DataVisualization](/jupyter_notebooks/02%20-%20DataVisualization.ipynb).
- Background images were taken from [Pixabay](https://pixabay.com/).

[Table Of Contents](#table-of-contents)


