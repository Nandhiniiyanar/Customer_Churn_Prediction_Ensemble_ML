# Telecom Customer Churn Prediction using Ensemble Machine Learning

## Project Overview

This project aims to predict customer churn in the telecommunications industry using ensemble machine learning algorithms. It analyzes customer subscription records and identifies customers who are likely to discontinue their services.

Two machine learning models, Random Forest and XGBoost, are trained and evaluated to identify the best-performing model.

## Problem Statement

Customer churn is a major challenge in the telecommunications industry. Losing customers can affect business growth and revenue. This project uses machine learning techniques to predict customer churn based on customer demographics, subscription details, services, and payment information.

## Objectives

- Predict whether a customer will leave or continue using telecom services.
- Perform data preprocessing and feature encoding.
- Handle class imbalance using SMOTE.
- Train Random Forest and XGBoost classifiers.
- Evaluate model performance using ROC-AUC.
- Perform 5-fold stratified cross-validation.
- Identify the top 10 important features influencing predictions.

## Dataset

**Dataset:** Telco Customer Churn

**Source:** IBM Telco Customer Churn Dataset

**Dataset link:** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

- Total Records: 7,043
- Total Columns: 21
- Target Variable: Churn
- Target Classes: Yes and No

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SMOTE
- Joblib
- Google Colab

## Machine Learning Workflow

1. Dataset loading and exploration.
2. Data cleaning and preprocessing.
3. Conversion of TotalCharges into numeric format.
4. Categorical feature encoding using OneHotEncoder.
5. Train-test splitting (80:20).
6. Class balancing using SMOTE.
7. Random Forest model training.
8. XGBoost model training.
9. Model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC.
10. 5-fold stratified cross-validation.
11. Feature importance analysis.
12. Final model selection and saving.

## Model Performance

| Evaluation Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 77.50% | 79.70% |
| ROC-AUC | 83.36% | 84.30% |
| Mean CV ROC-AUC | 83.88% | 84.17% |

**Best Model:** XGBoost Classifier

**Best ROC-AUC Score:** 84.30%

Both models achieved the required ROC-AUC target of above 80%.

## Top 10 Important Features

The XGBoost model identified the following important features:

1. Contract - Month-to-month
2. Online Security - No
3. Payment Method - Electronic check
4. Tech Support - No
5. Paperless Billing - Yes
6. Internet Service - Fiber optic
7. Online Backup - No
8. Paperless Billing - No
9. Partner - Yes
10. Internet Service - DSL

## Project Files

```text
Customer_Churn_Prediction/
│
├── Customer_Churn_Prediction.ipynb
├── Telco-Customer-Churn.csv
├── best_churn_model.pkl
├── churn_preprocessor.pkl
├── requirements.txt
└── README.md
```

## Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone this repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Open the project folder.

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Open `Customer_Churn_Prediction.ipynb` in Google Colab or Jupyter Notebook.

5. Upload the dataset if required.

6. Run the notebook cells sequentially to reproduce the analysis and model training.

## Model Files

- `best_churn_model.pkl`: Saved trained XGBoost model.
- `churn_preprocessor.pkl`: Saved OneHotEncoder preprocessing transformer.

## Conclusion

This project demonstrates how ensemble machine learning techniques can be used to predict customer churn in the telecommunications industry.

Among the two evaluated models, XGBoost achieved the highest test ROC-AUC score of 84.30%, while Random Forest achieved 83.36%.

The results demonstrate the usefulness of machine learning for identifying customers who may discontinue telecom services.

## Author

**Nandhini I**

Machine Learning | Data Science | Python
