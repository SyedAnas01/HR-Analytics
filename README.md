
#AI-Driven Employee Engagement and Performance Analytics
## Overview

This project aims to predict employee attrition in an organization using the IBM Employee Attrition dataset. By employing data preprocessing techniques, machine learning models, and explainable AI methods, we analyze the factors influencing employee turnover and provide actionable insights to reduce attrition rates. This project aligns with the objectives of modernizing data infrastructure and generating insights, crucial for the People Analytics function at Microsoft.

### Keywords
- Data Analysis
- ETL (Extract, Transform, Load)
- Machine Learning
- Predictive Modeling
- Employee Engagement
- People Analytics
- Data Infrastructure
- Insights Generation

## Table of Contents
1. [Installation](#installation)
2. [Data Description](#data-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Modeling Techniques](#modeling-techniques)
6. [Evaluation Metrics](#evaluation-metrics)
7. [SHAP Analysis](#shap-analysis)
8. [Visualizations](#visualizations)
9. [Conclusions and Recommendations](#conclusions-and-recommendations)
10. [Future Work](#future-work)

## Installation

To run this project, ensure you have the following software installed:

- Python 3.7 or higher
- Anaconda or virtual environment (recommended)

### Required Packages
You can install the required packages using pip or conda:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap plotly
```

Or using conda:

```bash
conda install pandas numpy scikit-learn xgboost imbalanced-learn shap plotly
```

## Data Description

The dataset used for this project is the IBM Employee Attrition dataset, which contains various features related to employees, including:

- `Age`: Age of the employee
- `DistanceFromHome`: Distance the employee lives from work
- `JobRole`: Role of the employee in the organization
- `EducationField`: Educational background
- `MaritalStatus`: Marital status of the employee
- `Attrition`: Target variable indicating whether the employee has left the organization (Yes/No)

The dataset is stored in `IBM-Employee-Attrition.csv`.

## Data Preprocessing

Data preprocessing steps include:

1. **Handling Missing Values**: Filled missing values using forward fill method.
2. **One-Hot Encoding**: Categorical variables are converted into numerical format using one-hot encoding.
3. **Feature Scaling**: Numerical features are scaled using `StandardScaler` to normalize the data.
4. **Train-Test Split**: The dataset is split into training and testing sets (80% train, 20% test).
5. **Handling Class Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the classes in the training data.

### Keywords
- Data Cleaning
- Data Transformation
- Categorical Encoding
- SMOTE

## Exploratory Data Analysis (EDA)

Initial data exploration was conducted to understand patterns and relationships within the data. Key visualizations were created to showcase insights on employee attrition rates based on various features.

### Keywords
- Data Exploration
- Insights
- Visualizations

## Modeling Techniques

The following modeling techniques are implemented:

- **XGBoost Classifier**: An advanced gradient boosting model used for classification tasks.

### Hyperparameters
Hyperparameters for the XGBoost model are set as follows:

```python
gbm_model = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
```

## Evaluation Metrics

The model is evaluated using the following metrics:

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **AUC (Area Under the Curve)**: Measures the model's ability to distinguish between classes.
- **Classification Report**: Provides precision, recall, and F1-score for each class.
- **Confusion Matrix**: A table used to describe the performance of a classification model.

## SHAP Analysis

SHAP (SHapley Additive exPlanations) values are calculated to interpret the model's predictions. The summary plot provides insights into which features most influence employee attrition.

## Visualizations

Visualizations include:

1. **Confusion Matrix**: A heatmap displaying true positives, false positives, true negatives, and false negatives.
2. **ROC Curve**: A graph showing the trade-off between the true positive rate and false positive rate.
3. **Feature Importance**: A bar plot showing the importance of each feature in predicting attrition.
4. **Distribution of Predicted Probabilities**: A histogram representing the predicted probabilities of attrition.

### Keywords
- Data Visualization
- Interactive Dashboards
- Plotly

## Conclusions and Recommendations

The analysis reveals key factors contributing to employee attrition. Recommendations for reducing attrition include:

- Implementing targeted retention strategies based on high-risk employee characteristics.
- Enhancing employee engagement programs, especially for roles with higher attrition rates.

## Future Work

- **Integration with Real-Time Data Sources**: Consider connecting the model to live data streams for continuous monitoring and prediction.
- **Implementation of Explainable AI**: Incorporate SHAP or LIME for explainability of model predictions.
- **Deployment of Interactive Dashboards**: Develop dashboards using Power BI or Tableau to visualize insights for stakeholders.
