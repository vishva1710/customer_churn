#customer churn prediction

#project overview:
This project predicts whether a customer is likely to leave the bank using logistic Regression

#dataset:
customer information
credit score
age
balance
Geography
Gender
EstimatedSalary

10,000 rows,13 columns

#project workflow:
Data loading
Data cleaning
EDA
Encoding
Train-Test-split
Smote
StandardScaling
try multiple algorithms
    *logistic Regression
    *DecisionTreeClassifier
    *RandomForestClassifier

#best model performance is LogisticRegression:

#model performance in logistic Regression:
Accuracy:72.04%
Precision:38.85%
Recall:70.82%

#RandomForest achived 80.2% Recall after SMOTE balancing:

#Key insights:
Age showed positive correlation with churn
Germany customers had hiher churn rates
Active members were likely to churn

#pickle is used to save trained machine learning model:

#Author:
Vishva
