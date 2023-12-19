# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 18:03:16 2023

@author: Marco
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os
import joblib

"""
Data Exploration
"""

url = 'https://raw.githubusercontent.com/MahsaDorri/DataWarehouse/main/Bicycle_Thefts_Open_Data.csv'
data = pd.read_csv(url)
print(data.columns.values)
print(data.shape)
print(data.describe())
print(data.dtypes) 
print(data.head(5))

"""
Handling Missing Data and Dropping of Irrelevant Features
"""
#Drop Irrelevant features
columns_to_drop = ['X', 'Y', 'OBJECTID','EVENT_UNIQUE_ID','OCC_YEAR','OCC_MONTH','BIKE_MODEL','OCC_DOW','OCC_DAY','OCC_DOY','REPORT_YEAR','REPORT_MONTH','REPORT_DOW','REPORT_DAY','REPORT_DOY','LOCATION_TYPE','HOOD_158','HOOD_140','NEIGHBOURHOOD_140','LONG_WGS84','LAT_WGS84']
data.drop(columns=columns_to_drop, inplace=True)
# Drop rows where the 'STATUS' column is 'Unknown'
data_clean = data[data['STATUS'] != 'UNKNOWN']
# Replace 'STOLEN' with 0 and 'RECOVERED' with 1 in the 'STATUS' column
data_clean.loc[data_clean['STATUS'] == 'STOLEN', 'STATUS'] = 0
data_clean.loc[data_clean['STATUS'] == 'RECOVERED', 'STATUS'] = 1
# Fill missing values 
data_clean = data_clean.copy()
data_clean['BIKE_MAKE'].fillna('Unknown', inplace=True)
data_clean['BIKE_SPEED'].fillna(data_clean['BIKE_SPEED'].median(), inplace=True)
data_clean['BIKE_COST'].fillna(data_clean['BIKE_COST'].mean(), inplace=True)
most_frequent_color = data_clean['BIKE_COLOUR'].mode()[0]
data_clean['BIKE_COLOUR'].fillna(most_frequent_color, inplace=True)
print(data_clean.columns.values)
print(data_clean.isnull().sum())

"""
Normalization of Catergorical and Numerical
"""
data_normalized = data_clean.copy()
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
# Seperate feature from categorical and numerical
categorical_columns = []
numerical_columns = []
for col, col_type in data_normalized.dtypes.items():
    if col_type == 'object':
        categorical_columns.append(col)
    else:
        numerical_columns.append(col)
# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Apply LabelEncoder to categorical columns
for col in categorical_columns:
    if col in data_normalized.columns:
        data_normalized[col] = label_encoder.fit_transform(data_normalized[col].astype(str)) 
# Perform Min-Max normalization on numerical columns
scaler = MinMaxScaler()
data_normalized[numerical_columns] = scaler.fit_transform(data_normalized[numerical_columns])
print(data_normalized.dtypes) 

"""
Handling Imbalance class & Split and Train
"""
# Assuming 'processed_data' contains the preprocessed DataFrame
X = data_normalized.drop('STATUS', axis=1)
y = data_normalized['STATUS']
# Check for NaN values in 'y' and drop corresponding rows in 'X' and 'y'
nan_indices_y = y[y.isnull()].index
nan_indices_X = X[X.isnull().any(axis=1)].index
nan_indices = nan_indices_y.union(nan_indices_X)
X.drop(index=nan_indices, inplace=True)
y.drop(index=nan_indices, inplace=True)
# Apply SMOTE after handling NaN values in 'y'
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
# Split the resampled data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
# Check class counts after balancing
class_counts = pd.Series(y_resampled).value_counts()
print(class_counts)

"""
Feature Selection
"""
# Assuming X_train and y_train are the preprocessed training data after SMOTE
# Initialize the model (e.g., LinearRegression)
estimator = DecisionTreeClassifier(random_state=42)
# Initialize RFE and fit it to your data
n_features_to_select = 5
rfe = RFE(estimator, n_features_to_select=n_features_to_select)
X_rfe = rfe.fit_transform(X_train, y_train)
# Get the selected features
selected_features = X_train.columns[rfe.support_]
# Print the selected features
print("Selected Features:")
print(selected_features)

"""
Predictive model building
"""
# Initialize the Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Fit the model on the training data
decision_tree_model.fit(X_train, y_train)

# Predict on the test data
y_pred_dt = decision_tree_model.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy: {accuracy_dt:.2f}")

# Classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# ROC Curve
y_pred_proba_dt = decision_tree_model.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_pred_proba_dt)
roc_auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree ROC Curve (AUC = {roc_auc_dt:.2f})')
plt.plot([0, 1], [0, 1], 'r--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree - Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
print(f"ROC AUC: {roc_auc_dt:.2f}")


"""
Serialize the model
"""
# Get the current directory path
current_directory = os.path.dirname(os.path.abspath(__file__))  # Assuming this code is in a script or module
# Serialize and save the model
joblib.dump(decision_tree_model, os.path.join(current_directory, 'model_lr.pkl'))
print("Model dumped!")
# Serialize and save the model columns as an object
model_columns = list(X.columns)
joblib.dump(model_columns, os.path.join(current_directory, 'model_columns.pkl'))
print("Model columns dumped!")
