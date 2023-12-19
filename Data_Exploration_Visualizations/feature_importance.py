# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier

# Set path for the data
dataPath = "C:/Users/parth/Downloads/"
filename = 'Bicycle_Thefts_Open_Data.csv'


data = os.path.join(dataPath, filename)

# Load data from a CSV file
data_group3 = pd.read_csv(data)

# Replace 'NSA' and handle missing values
data_group3.replace('NSA', np.nan, inplace=True)

# Handle missing values for numerical columns
numerical_cols = data_group3.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    data_group3[col].fillna(data_group3[col].mean(), inplace=True)

# Label encoding for categorical variables
label_encoder = LabelEncoder()
categorical_cols = data_group3.select_dtypes(include=['object']).columns.tolist()

# Exclude the target variable 'STATUS' from the categorical columns list
if 'STATUS' in categorical_cols:
    categorical_cols.remove('STATUS')

# Encode categorical columns
for col in categorical_cols:
    data_group3[col] = label_encoder.fit_transform(data_group3[col].astype(str))

# Exclude non-informative features
non_informative_features = ['X', 'Y', 'OBJECTID', 'EVENT_UNIQUE_ID']

# Prepare data for modeling
X = data_group3.drop(non_informative_features + ['STATUS', 'OCC_DATE', 'REPORT_DATE'], axis=1, errors='ignore')
y = label_encoder.fit_transform(data_group3['STATUS'])

# Check if there are any NaN values left
if X.isnull().any().any():
    raise ValueError("NaN values are still present in the data.")

# Initialize the ExtraTreesClassifier
model = ExtraTreesClassifier()

# Fit the model
model.fit(X, y)

# Output the feature importances
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
print(feat_importances)

# Plot graph of feature importances for better visualization
plt.figure(figsize=(12, 8))  # Increase figure size
feat_importances.nlargest(len(X.columns)).plot(kind='barh')  # Plot all features
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()  # Adjust layout to fit all labels
plt.show()
