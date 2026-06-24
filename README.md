**Customer Churn Prediction**

Problem Statement:

**Predicting customer churn using machine learning models to classify customers as Exited or Retained**

Dataset:

*Source:Churn_Modelling.csv(kaggle),
*Total Records:10,000 customers,
*Features:11 columns(after dropping unnecassary ones).

**Features used**:

Feature:CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,IsActiveMember,Estimatedsalary,
Exited-Target Column  1-churned,0-Retained.

**Dropped Columns**:

RowNumbers,CustomerId,Surname-Not useful for prediction.

**Project Workflow**:

import necessary libraries,
Data loading,
Data exploration,
EDA(boxplot-outliers,Scatterplot-for realtionship between two features,histplot-ditribution,heatmap-for correlation),
Data Preprocessing,
Feature engineering,
Class imbalance using smote analysis,
Train-test-split,
model trained to try   multiple,algorithms:RandomForestClassifier,DecisionTreeClassifier,XGBClassifier.

**Results**:

Model DecisionTree accuracy:0.7666,Precision:0.4475,Recall:0.7203,F1score:0.5520,
Model RandomForest accuracy:0.8156,Precision:0.5287,Recall:0.6680,F1score:0.5902,
Model XGBoost accuracy:0.8144,Precision:0.5279,Recall:0.6278,F1score:0.5735.

**ConfusionMatrix**:

Model DecisonTree TN:1561,FP:442,FN:139,TP:358,
Model RandomForest TN:1707,FP:296,FN:165,TP:332,
Model XGBoost TN:1724,FP:279,FN:185,TP:312.

**Best Model**:(RandomForest):

*Highest F1score(0.590)-best balance between Precison and Recall,
*Good Accuracy(81.56%),
*Recall(0.668)-catches 67% of actual churners.

**Key Observations**:

*Dataset is Imbalanced-SMOTE applied to fix it,
*Age column has outliers(60-92 range) but not removed-valid customer data,
*CreditScore distribution is slightly left skewed,
*Age distribution is slightly right skewed,
XGBoost had very high Recall (0.99) but extremeky low Precision-too many false alarms.

**Save Model**:

import pickle
best_model=models["RandomForest"]
pickle.dump(best_model,open("model.pkl","wb"))
model=pickle.load(open("model.pkl","rb"))

**Requirements**:

numpy,
pandas,
matplotlib,
seaborn,
scikit-learn,
imbalanced-learn,
xgboost.


