# Restaurant Tip Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Scikit--learn%20%7C%20Seaborn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This project explores the use of various machine learning regression models to predict the tip amount a customer will give at a restaurant. Using the classic "tips" dataset, we preprocess the data, train multiple models, and evaluate their performance to find the most accurate predictor.

##  Table of Contents
- [Project Goal](#-project-goal)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Model Evaluation](#3-model-evaluation)
- [Results Summary](#-results-summary)
- [How to Run This Project](#-how-to-run-this-project)
- [Libraries Used](#-libraries-used)


## Project Goal

The primary objective of this project is to build and evaluate several regression models to predict the `tip` amount based on various features of a restaurant bill. This involves a complete machine learning workflow from data preparation to model comparison.

##  Dataset

This project uses the **"tips"** dataset, which is readily available in the Seaborn library. It contains records of tips given by customers in a restaurant over a period of a few months.

The features in the dataset are:
-   `total_bill`: The total cost of the meal in dollars.
-   `tip`: The tip amount given in dollars (This is our **target variable**).
-   `sex`: The gender of the person paying the bill (Male/Female).
-   `smoker`: Whether the party included smokers (Yes/No).
-   `day`: The day of the week (Thur, Fri, Sat, Sun).
-   `time`: The time of the meal (Lunch/Dinner).
-   `size`: The number of people in the party.

##  Methodology

The project follows a standard machine learning pipeline:

### 1. Data Preprocessing

Before training the models, the data is carefully prepared to be suitable for machine learning algorithms.
1.  **Feature and Target Separation**: The dataset is split into features (`X`) and the target variable (`y`, which is `tip`).
2.  **Categorical Data Encoding**: Categorical features like `sex`, `smoker`, `day`, and `time` cannot be used directly. We use **One-Hot Encoding** (`pd.get_dummies`) to convert them into a numerical format. To prevent multicollinearity, `drop_first=True` is used, which removes the first category of each feature.
3.  **Feature Scaling**: The numerical features (`total_bill`, `size`, and the newly encoded features) are scaled using `StandardScaler` from Scikit-learn. This standardizes the features to have a mean of 0 and a standard deviation of 1, which helps many algorithms (like SVR and KNN) perform better.
4.  **Train-Test Split**: The processed dataset is split into a training set (80%) and a testing set (20%) to evaluate the models on unseen data.

### 2. Model Training

Four different regression models were trained on the preprocessed training data:
1.  **Linear Regression**: A baseline model that assumes a linear relationship between the features and the target.
2.  **K-Nearest Neighbors (KNN) Regressor**: A non-parametric model that predicts based on the average tip of the `k` nearest data points in the feature space (`k=5` was used).
3.  **Random Forest Regressor**: An ensemble model that builds multiple decision trees and merges their predictions to produce a more accurate and stable result (`n_estimators=500`).
4.  **Support Vector Regressor (SVR)**: A powerful model that finds a hyperplane to best fit the data, effective in high-dimensional spaces.

### 3. Model Evaluation

The performance of each model is evaluated on the test set using the following standard regression metrics:
-   **Mean Absolute Error (MAE)**: The average of the absolute differences between the predicted and actual values.
-   **Mean Squared Error (MSE)**: The average of the squared differences between the predicted and actual values. It penalizes larger errors more heavily.
-   **Root Mean Squared Error (RMSE)**: The square root of the MSE. It provides an error metric in the same unit as the target variable (dollars), making it highly interpretable.

Additionally, **10-fold Cross-Validation** was performed to ensure the model's performance is robust and not just a result of a favorable train-test split.

##  Results Summary

The performance of the four models on the test set is summarized below. The **Root Mean Squared Error (RMSE)** is the primary metric for comparison, as it is in the same unit as the tip amount.

| Model                     | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) |
|---------------------------|---------------------------|--------------------------|--------------------------------|
| Linear Regression         | 0.708                     | 0.894                    | 0.945                          |
| K-Nearest Neighbors (k=5) | 0.751                     | 0.946                    | 0.973                          |
| **Random Forest**         | **0.705**                 | **0.802**                | **0.896**                      |
| Support Vector Regressor  | 0.736                     | 0.968                    | 0.984                          |

### Conclusion
Based on the results, the **Random Forest Regressor** is the best-performing model with the lowest RMSE of **$0.896**. This means, on average, its predictions are off by about 90 cents, which is a strong result for this dataset.

##  How to Run This Project

### Prerequisites
-   Python 3.7+
-   Jupyter Notebook or JupyterLab

## Libraries Used
-   **NumPy**: For numerical operations.
-   **Pandas**: For data manipulation and analysis.
-   **Matplotlib & Seaborn**: For data visualization.
-   **Scikit-learn**: For machine learning, including preprocessing, model training, and evaluation.
