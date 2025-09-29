# Forest Cover Type Prediction

## Project Overview

This project uses cartographic variables to classify forest cover types using machine learning techniques. The goal is to develop a robust model that can accurately predict forest cover types to assist in environmental conservation and forest management efforts.

## Dataset

The dataset contains cartographic variables for forest cover type classification. The data includes:
- Elevation, aspect, slope information
- Distance to water features, roads, and fire points
- Soil type and wilderness area information
- Target variable: 7 different forest cover types

## Project Structure

```
project/
│   README.md
│   requirements.txt
│
└───data/
│   │   train.csv
│   │   test.csv
│   │   covtype.info
│
└───notebook/
│   │   EDA.ipynb
│
└───scripts/
│   │   preprocessing_feature_engineering.py
│   │   model_selection.py
│   │   predict.py
│
└───results/
    │   plots/
    │   test_predictions.csv
    │   best_model.pkl
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Exploratory Data Analysis**: Run the Jupyter notebook `notebook/EDA.ipynb` to explore the dataset
2. **Model Training**: Execute `python scripts/model_selection.py` to train and select the best model
3. **Prediction**: Run `python scripts/predict.py` to make predictions on the test set

## Model Performance

### Training Results
- Best Model: [To be filled after training]
- Training Accuracy: [To be filled - must be < 0.98]
- Cross-Validation Score: [To be filled]

### Test Results
- Test Accuracy: [To be filled - target > 0.65]
- Final Model: [Model name and parameters]

## Features Engineered

- Distance to hydrology: sqrt((Horizontal_Distance_To_Hydrology)² + (Vertical_Distance_To_Hydrology)²)
- Road-Fire distance difference: Horizontal_Distance_To_Fire_Points - Horizontal_Distance_To_Roadways
- Additional domain-specific features based on EDA insights

## Models Evaluated

1. Gradient Boosting
2. K-Nearest Neighbors (KNN)
3. Random Forest
4. Support Vector Machine (SVM)
5. Logistic Regression

## Key Findings

[To be filled after analysis]

## Contributors

Data Scientist - Environmental Conservation Agency
