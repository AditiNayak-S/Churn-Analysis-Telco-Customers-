# Churn-Analysis-Telco-Customers-
Customer Churn Analysis – Telco Dataset
Project Objective

The primary objective of this project is to analyze customer behavior within a telecommunication dataset and predict which customers are likely to churn (leave the service). The insights gained from this analysis can help the business take proactive measures to retain valuable customers.

Tools Used

Python
pandas
numpy
matplotlib
seaborn
scikit-learn
sqlite3

Data Loading and Initial Exploration

Dataset Source: Telco Customer Churn (Kaggle)
Dataset Shape: 7043 rows, 21 columns

Columns:
customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn

Data Types:
Object (categorical), int64 (integers), float64 (numerical)

Churn Distribution:
No: 5174 customers
Yes: 1869 customers

The dataset is imbalanced, with significantly fewer churned customers.

Exploratory Data Analysis (EDA)

Customer Churn Distribution
A count plot showed a clear imbalance in the churn target variable.

Tenure vs Churn
Customers who churn generally have lower tenure compared to non-churned customers.

Monthly Charges vs Churn
Churning customers tend to have higher monthly charges.

Contract Type vs Churn
Customers on month-to-month contracts exhibit much higher churn rates than those on one-year or two-year contracts.

Data Preprocessing

TotalCharges Conversion
The TotalCharges column was converted from object to numeric using errors='coerce'.

Handling Missing Values
Rows containing NaN values created during conversion were dropped.

Feature Engineering
The customerID column was removed as it is an identifier.
The Churn target variable was encoded as 1 for Yes and 0 for No.

Categorical Feature Encoding
Categorical features were converted using one-hot encoding with drop_first=True.

Numerical Feature Scaling
SeniorCitizen, tenure, MonthlyCharges, and TotalCharges were scaled using StandardScaler.

Data Splitting
The dataset was split into training and testing sets using an 80/20 ratio with stratification to preserve churn distribution.

Model Training and Evaluation
Logistic Regression (Baseline)

Confusion Matrix
[[918, 115],
[161, 213]]

Performance Metrics
Precision (Churn): 0.65
Recall (Churn): 0.57
Accuracy: 0.80

Logistic Regression (Class-Weighted and Threshold Adjusted)

Class-Weighted Logistic Regression (Default Threshold)
Confusion Matrix
[[724, 309],
[77, 297]]

Recall (Churn): 0.79
Precision (Churn): 0.49
Accuracy: 0.73

Threshold Adjusted Logistic Regression (Threshold = 0.4)
Confusion Matrix
[[637, 396],
[48, 326]]

Recall (Churn): 0.87
Precision (Churn): 0.45
Accuracy: 0.68

ROC-AUC Score: 0.84

Random Forest Classifier

Model Parameters
n_estimators = 300
random_state = 42
class_weight = balanced

Confusion Matrix
[[925, 108],
[192, 182]]

Performance Metrics
Precision (Churn): 0.63
Recall (Churn): 0.49
Accuracy: 0.79

Feature Importance and Coefficients

Logistic Regression Coefficients

Top Positive Coefficients (Higher Churn Likelihood)
InternetService_Fiber optic: 1.149
TotalCharges: 0.611
PaymentMethod_Electronic check: 0.412
StreamingTV_Yes: 0.391
StreamingMovies_Yes: 0.376

Top Negative Coefficients (Lower Churn Likelihood)
Contract_Two year: -1.452
tenure: -1.261
Contract_One year: -0.786
MonthlyCharges: -0.503
OnlineSecurity_Yes: -0.346

Random Forest Feature Importance (Top Features)
TotalCharges: 0.178
tenure: 0.165
MonthlyCharges: 0.152
Contract_Two year: 0.059
InternetService_Fiber optic: 0.040

Risk Categorization and Cost Analysis

Based on predicted churn probabilities from the class-weighted Logistic Regression model:

High Risk (Probability ≥ 0.7): 378 customers
Medium Risk (0.4 ≤ Probability < 0.7): 344 customers
Low Risk (Probability < 0.4): 685 customers

Cost Assumptions
False Negative (missed churner): 5
False Positive (unnecessary retention effort): 1

Total misclassification cost using a 0.4 threshold: 636

Model Comparison

Model | Churn Recall | Churn Precision | Accuracy
Logistic Regression (Baseline) | 0.57 | 0.65 | 0.80
Logistic Regression (Class-Weighted) | 0.79 | 0.49 | 0.73
Logistic Regression (Threshold = 0.4) | 0.87 | 0.45 | 0.68
Random Forest | 0.49 | 0.63 | 0.79

The class-weighted Logistic Regression model with a threshold of 0.4 provides the best recall for identifying potential churners.

Database Integration

The processed data was stored in a SQLite database named churn_analysis.db.

Tables Created
customers
services
billing

Note
The churn column in the billing table was stored as Yes and No instead of numeric values, leading to incorrect churn rate calculations in SQL. This column should be re-encoded to 0 and 1 for accurate aggregation.

Conclusion and Recommendations

This project successfully identified key factors influencing customer churn and built predictive models to support customer retention strategies.

Key Findings
Customers with month-to-month contracts, higher monthly charges, and shorter tenure are more likely to churn.
Fiber optic internet service and electronic check payment methods are associated with higher churn.
Longer contracts, longer tenure, and security or support services reduce churn likelihood.

Recommendations
Focus retention efforts on high-risk and medium-risk customers.
Encourage customers to move from month-to-month contracts to longer-term contracts.
Investigate service quality issues related to fiber optic and streaming services.
Promote stable payment methods and incentives.
Engage customers early, especially those with increasing monthly charges or short tenure.
