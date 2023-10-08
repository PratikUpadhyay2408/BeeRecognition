# Bee Subspecies Classification Project

## Overview

Brief description of the project and its objectives. Include what the project aims to achieve and why it's important.

## Repository Structure


- **BeeData** 
  - Contains data for training the bee recognition model.

- **Templates** 
  - HTML templates for the Flask web application.

- **Results** 
  - contains the complete trained model ( finalmodel.h5 ) and the weights have additionally been saved separately in cnn_model.h5
  - images used in the readme and some  metrics for model performance.

- **Bee_Image_Recognition.py** 
  - Python file containing code for the Deployable Bee Recognition Flask App.

- **Data_Processing.py** 
  - contains code for processing images and creating datasets.

- **Model.py** 
  - contains the code for the bee recognition Machine Learning model.
  - Also has a classifier using vgg16 backbone with custom top implemented.  

- **Predict.py** 
  - contains the code for making predictions using the trained model.

- **Training.py** 
  - Python file containing code for training the bee recognition model.


## Data

The dataset comprises images of honey bees from four distinct subspecies: **_Russian Honey Bee_**, **_Italian Honey Bee_**, **_Carniolan Honey Bee_** and **_Mixed Local Stock_**. Each subspecies is represented by specific observations. Below are the images of the bees from the respective subspecies:

<div style="display: flex; justify-content: space-around; align-items: flex-start;">
    <img src="https://github.com/PratikUpadhyay2408/BeeRecognition/blob/main/results/russian-bee.png" alt="Russian Bee" width="200" height="150">
    <img src="https://github.com/PratikUpadhyay2408/BeeRecognition/blob/main/results/italian.jpg" alt="Italian Honey Bee" width="200" height="150">
    <img src="https://github.com/PratikUpadhyay2408/BeeRecognition/blob/main/results/carnica.jpg" alt="Carniolan Honey Bee" width="200" height="150">
   <img src="https://github.com/PratikUpadhyay2408/BeeRecognition/blob/main/results/mixed.jpg" alt="Mixed Local Stock" width="200" height="150">
</div> 
from left to right <em>Russian Honey Bee</em>, <em>Italian Honey Bee</em>, <em>Carniolan Honey Bee</em>, <em>Mixed Local Stock</em>


### Sample Data
<img src="https://github.com/PratikUpadhyay2408/BeeRecognition/blob/main/results/bee_species.png" alt="images from actual data" >


| File               | Date    | Time  | Location           | Zip Code | Subspecies         | Health            | Pollen Carrying | Caste  |
|--------------------|---------|-------|--------------------|----------|--------------------|--------------------|-----------------|--------|
| 041_066.png        | 8/28/18 | 16:07 | Alvin, TX, USA     | 77511    | Russian Honey Bee  | Hive being robbed  | FALSE           | Worker |
| 041_072.png        | 8/28/18 | 16:07 | Alvin, TX, USA     | 77511    | Italian Honey Bee  | Hive being robbed  | FALSE           | Worker |
| 041_073.png        | 8/28/18 | 16:07 | Alvin, TX, USA     | 77511    | Russian Honey Bee  | Hive being robbed  | FALSE           | Worker |
| 041_067.png        | 8/28/18 | 16:07 | Alvin, TX, USA     | 77511    | Italian Honey Bee  | Hive being robbed  | FALSE           | Worker 

In this dataset:

- **File**: Represents the filename of the bee image.
- **Date**: Indicates the date of the observation.
- **Time**: Specifies the time of the observation.
- **Location**: Provides the specific location where the observation took place, including city and state (Alvin, TX, USA).
- **Zip Code**: Contains the ZIP code of the observation location (77511).
- **Subspecies**: Indicates the subspecies of the observed bee, including **_Russian Honey Bee_**, **_Italian Honey Bee_**, and **_Mixed Local Stock_**.
- **Health**: Describes the health status of the observed bee, with this sample indicating the hive being robbed.
- **Pollen Carrying**: Indicates whether the bee is carrying pollen (FALSE in this case, denoting no pollen carrying).
- **Caste**: Represents the caste of the observed bee, with all instances belonging to the worker caste.


## Model Architecture
<img src="https://github.com/PratikUpadhyay2408/BeeRecognition/blob/main/results/archi_bee_classification.png" alt="model arch" >

**Input Layer:**

The model starts with a Conv2D layer having 6 filters (3x3 each) and a ReLU activation function. It takes input images with a shape defined by `self.input_shape`.

**Pooling Layer:**

A MaxPooling2D layer with a pool size of 2x2 follows the first convolutional layer. It reduces the spatial dimensions, making the network more efficient.

**Second Convolutional Layer:**

Another Conv2D layer with 12 filters (3x3 each) and ReLU activation is added, further refining the learned features.

**Flattening:**

The output from the second convolutional layer is flattened into a one-dimensional vector using the Flatten layer, preparing it for input into the densely connected layers.

**Dense (Fully Connected) Layer:**

A Dense layer with `self.output_size` neurons and softmax activation produces the final class probabilities.

*Note: We initially attempted to use a VGG16 architecture, but it had too many layers, leading to overfitting issues.*

## Results

<div style="display: flex; justify-content: space-around; align-items: flex-start;">
    <img src="https://github.com/PratikUpadhyay2408/BeeRecognition/blob/main/results/model_accuracy.png" alt="model accuracy" height=300>
    <img src="https://github.com/PratikUpadhyay2408/BeeRecognition/blob/main/results/model_loss.png" alt="model loss" height=300>
</div> 

<img src="https://github.com/PratikUpadhyay2408/BeeRecognition/blob/main/results/test_accuracy.png" alt="test accuracy" width=400>

Present the results of the experiments conducted using the model. Include metrics, charts, and any other visualizations that help explain the performance of the classification model.

