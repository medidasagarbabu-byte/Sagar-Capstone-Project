
# Titanic Analytics Assessment

## Project Overview

This project analyzes the Titanic dataset using Python and machine learning.
The analysis includes exploratory data analysis, classification, class
imbalance handling, Random Forest tuning, regression, and residual analysis.

## Exploratory Data Analysis

The dataset was explored using descriptive statistics and visualizations.
The analysis includes distributions, survival patterns, relationships between
features, and correlation analysis.

## Classification

The following classification models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

The classification models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Class Imbalance

Random Forest performance was compared using:

- Baseline Random Forest
- Random Forest with balanced class weights
- Random Forest with SMOTE

SMOTE was applied only to the training data to avoid data leakage.

The SMOTE Random Forest achieved the strongest overall results among the
imbalance approaches tested.

## Random Forest Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used to tune the Random Forest
classifier.

The best parameters were:

- n_estimators: 200
- max_depth: None
- min_samples_split: 2
- min_samples_leaf: 2

The Random Forest OOB score was 0.8270.

## Regression

Fare was used as the regression target.

The following models were evaluated:

- Linear Regression
- Random Forest Regressor

Regression metrics:

- RMSE
- MAE
- R²

Residual analysis was performed and the Breusch-Pagan test was used to check
for heteroscedasticity.

The Breusch-Pagan test produced p-values below 0.05, providing evidence of
heteroscedasticity in the Linear Regression residuals.

## Folder Structure

analytics/
|
|-- titanic_analysis.ipynb
|-- titanic.csv
|-- README.md
|
|-- charts/
|   |-- Generated visualizations
|
|-- results/
|   |-- Model evaluation results
|
|-- models/
    |-- Saved machine learning models

## Main Model Artifacts

- logistic_regression_pipeline.pkl
- smote_preprocessor.pkl
- smote_random_forest.pkl
