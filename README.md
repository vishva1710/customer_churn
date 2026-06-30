**Customer Churn Prediction**:

Business Problem:
Bank Customers who are likely to leave the bank can be identified early using Machine Learning ,allowing the bank to take retention actions.

Dataset:
The dataset used is Churn_Modelling.csv, which contains information on 10,000 bank customers.
Original columns: RowNumber, CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited.

Target variable: Exited
0 → Customer stayed (No Churn),
1 → Customer left (Churn).

Class distribution:
0 (Stayed): 79.63%,
1 (Churned): 20.37%,
The dataset is imbalanced, which is addressed using SMOTE oversampling.

Project Workflow:

1. Data Preprocessing:
Drop unnecessary columns:
Rownumber,Customerid,Surname.

Check for null values:
No missing values were found across all columns (CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited).

Check for duplicate rows:

Encoding categorical features:
Geography was one-hot encoded using pd.get_dummies(), creating Geography_Germany and Geography_Spain columns (France is the reference category).

Gender was label encoded (Male = 1, Female = 0 or similar mapping).

After preprocessing, the dataset contains 12 columns and 10,000 rows, all non-null, with numeric dtypes.

2.Exploratory Data Analysis (EDA):
Gender distribution:
Male: 5,457,
Female: 4,543.

A pie chart was plotted to visualize the male vs. female ratio.

Distribution checks:
Histograms were plotted for numerical columns: CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary.

CreditScore distribution is approximately normal with a slight skew.
Age distribution is slightly right-skewed with outliers detected in a boxplot (values above ~60-65).
Outliers in Age were not removed since tree-based models are robust to them.

Scatter plot (Age vs Balance):

Used to visualize the relationship between Age and Balance, colored by the Exited label. Churned customers (orange) tend to cluster at higher ages.

Correlation heatmap:
A heatmap was generated to check feature correlations. Notable observations:

Age has a moderate positive correlation with Exited (~0.29),
Balance has some correlation with Exited,
NumOfProducts shows a moderate negative correlation with Balance (~-0.3).

Churn distribution:
A count plot confirmed the class imbalance — approximately 8,000 customers stayed vs. ~2,000 who churned.

3.Segregating Features and Target:

4.Train-Test Split-80:20

5. Handling Class Imbalance with SMOTE:
Since the dataset is imbalanced (80% stayed, 20% churned), SMOTE (Synthetic Minority Oversampling Technique) was applied to the training set to balance the classes.

Before SMOTE: 6,370 (stayed) vs 1,630 (churned),
After SMOTE: 6,370 vs 6,370 — perfectly balanced.

6. Feature Scaling:
Standard scaling was applied for Logistic Regression,
The scaler was saved using pickle for use during prediction on new data.

7. Model Training and Evaluation:
Baseline Model — Logistic Regression.

Comparing Multiple Algorithms:
Three models were trained and compared:
Model--Decision Tree:
Train Accuracy--0.7892,
Test Accuracy--0.7705,
Precision--0.7705,
Recall--0.7027,
F1 Score--0.5548.

Model--Random Forest:
Train Accuracy--0.8835,
Test Accuracy--0.7965,
Precision--0.5000,
Recall--0.6585,
F1 Score--0.5684.

Model--XGBoost:
Train Accuracy--0.8748,
Test Accuracy--0.8015,
Precision--0.5097,
Recall--0.6437,
F1 Score--0.5689,

Confusion matrices:
Decision Tree: [[1255, 338], [121, 286]]
Random Forest: [[1325, 268], [139, 268]]
XGBoost: [[1341, 252], [145, 262]]

Cross Validation (XGBoost):

8. Feature Importance:
Feature importance was extracted from the Random Forest model and plotted as a bar chart. The top contributing features were:
Age (most important, ~0.30),
NumOfProducts (~0.17),
Balance (~0.16),
Other features: CreditScore, EstimatedSalary, Gender, etc.

9.ROC Curve:
The ROC curve was plotted for the Random Forest model, showing strong discriminative ability with the curve rising steeply toward the top-left corner.

10.Saving the Model:
The best model (Random Forest) and the scaler were saved using pickle for future inference:

Results Summary:
The XGBoost model achieved the best test accuracy at 80.15%, while the Random Forest model had a strong cross-validation F1 average of ~82.45% and was selected as the final model. The combination of SMOTE oversampling, feature encoding, and standard scaling significantly improved prediction quality on the minority (churn) class.
