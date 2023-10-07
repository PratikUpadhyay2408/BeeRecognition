# Bee Subspecies Classification Project

## Overview

Brief description of the project and its objectives. Include what the project aims to achieve and why it's important.

## Repository Structure

/data
Contains datasets for training and testing the classification model.
/src
Contains the source code for the classification model.
/docs
Contains project documentation and related files.
/results
Stores the results of experiments and model evaluations.
/scripts
Contains helper scripts for data preprocessing, training, and evaluation.
bash


## Installation

```bash
# Clone the repository
git clone https://github.com/username/bee-subspecies-classification.git

# Navigate to the project directory
cd bee-subspecies-classification

# Install dependencies (if any)
pip install -r requirements.txt

# Example commands for training the model
python src/train.py --dataset /path/to/training/dataset --epochs 50 --batch_size 32

# Example commands for making predictions
python src/predict.py --model /path/to/trained/model --input /path/to/input/image.jpg

# Example commands for evaluating the model
python src/evaluate.py --model /path/to/trained/model --test_dataset /path/to/test/dataset
```
## Data

The dataset comprises images of honey bees from three distinct subspecies: **_Russian Honey Bee_**, **_Italian Honey Bee_**, and **_Mixed Local Stock_**. Each subspecies is represented by specific observations. Below are the images of the bees from the respective subspecies:

![Russian Bee](https://github.com/PratikUpadhyay2408/BeeRecognition/blob/main/results/russian-bee.jpg)
*Caption: Russian Honey Bee*

![Italian Bee](https://github.com/PratikUpadhyay2408/BeeRecognition/blob/main/results/italian.jpg)
*Caption: Italian Honey Bee*

![Mixed Bee](https://github.com/PratikUpadhyay2408/BeeRecognition/blob/main/results/mixed.jpg)
*Caption: Mixed Local Stock*

### Sample Data

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


## Model
Explain the classification model architecture, including layers, activation functions, and any other relevant details. If there are multiple iterations of the model, document the improvements and changes made in each version.

## Results
Present the results of the experiments conducted using the model. Include metrics, charts, and any other visualizations that help explain the performance of the classification model.

