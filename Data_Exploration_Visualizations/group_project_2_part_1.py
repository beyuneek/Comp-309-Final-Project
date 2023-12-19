# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 23:49:39 2023

@author: parth
"""

import pandas as pd
import os
path = "C:/Users/parth/Downloads/"
filename = 'Bicycle_Thefts_Open_Data.csv'
fullpath = os.path.join(path,filename)
data = pd.read_csv(fullpath,sep=';')


# View the first few rows of the dataset
print(data.head())

# Understanding the DataFrame's structure
print(data.info())

# Statistical summary of the dataset
print(data.describe())

#Column Descriptions

# Display data types of each column
print(data.dtypes)


#Ranges and Values


# Individual min and max computations for each numerical column
numerical_columns = data.select_dtypes(include=['int64', 'float64'])

print("Ranges for Numerical Columns:")
for column in numerical_columns:
    min_value = numerical_columns[column].min()
    max_value = numerical_columns[column].max()
    print(f"{column}: Min = {min_value}, Max = {max_value}")


# Unique values and their counts for categorical columns
categorical_columns = data.select_dtypes(include=['object'])
for column in categorical_columns:
    print(f"\nUnique Values in {column}:")
    print(data[column].value_counts())
    print("\n")
    
    
#    Statistical Assessments

# Descriptive statistics for numerical columns
print(data.describe())


#Distribution Analysis

import matplotlib.pyplot as plt
import seaborn as sns

# Histograms for numerical data
for col in numerical_columns.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()













# # Step 4: Missing Data Evaluations

# Importing necessary libraries
import pandas as pd               # For data manipulation and analysis
import missingno as msno          # For visualizing missing data
import matplotlib.pyplot as plt   # For creating plots and visualizations
import seaborn as sns             # For advanced data visualization

# Load the dataset from the specified file path
file_path = 'C:/Users/parth/Downloads/Bicycle_Thefts_Open_Data.csv'
data = pd.read_csv(file_path)

# Visualizing missing data in the dataset to understand the pattern of missingness
msno.matrix(data)
plt.show()

# Step 1 Handling Missing Values
# ---Loop through numerical columns to impute missing values with the median
for col in data.select_dtypes(include=['float64', 'int64']):
    data[col] = data[col].fillna(data[col].median())

# ---Removing rows with missing data in categorical columns
data.dropna(inplace=True)

# Step 2 ---Outlier Detection and Handling
# Applying the Interquartile Range (IQR) method to detect outliers in numerical columns
for col in data.select_dtypes(include=['float64', 'int64']):
    Q1 = data[col].quantile(0.25)               # Calculate the first quartile (Q1)
    Q3 = data[col].quantile(0.75)               # Calculate the third quartile (Q3)
    IQR = Q3 - Q1                               # Compute the Interquartile Range (IQR)
    lower_bound = Q1 - 1.5 * IQR                # Lower bound to identify outliers
    upper_bound = Q3 + 1.5 * IQR                # Upper bound to identify outliers
   
    # Option 2: Cap and floor outlier values to reduce their impact
    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

# Step 3 ---Correcting Data Types
# Example: Converting a date column from string to datetime format
# data['your_date_column'] = pd.to_datetime(data['your_date_column'])

# Step 4 ---Recheck for missing data after handling to ensure all missing values are addressed
print("\nMissing Data After Handling:")
print(data.isnull().sum())

# Visualizing the distribution of a numerical column with a boxplot
# Here, 'BIKE_COST' is visualized to understand its spread after handling outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['BIKE_COST'])
plt.title("Boxplot of Bike Cost After Handling Outliers")
plt.show()














#Step 5 : Graphs and Visualizations
#5.1 Visualizing Distributions
import pandas as pd

file_path = 'C:/Users/parth/Downloads/Bicycle_Thefts_Open_Data.csv'

# Try loading with a comma delimiter
data_comma = pd.read_csv(file_path, delimiter=',')

# Check columns
print("Columns with comma delimiter:", data_comma.columns)
# print("Columns with semicolon delimiter:", data_semicolon.columns)
# print("Columns with tab delimiter:", data_tab.columns)

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram for the 'BIKE_SPEED' column
plt.figure(figsize=(10, 6))
sns.histplot(data_comma['BIKE_SPEED'].dropna(), kde=True)  # Dropping NA values for visualization
plt.title('Histogram of Bike Speed')
plt.show()

# Boxplot for the 'BIKE_SPEED' column
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_comma, x='BIKE_SPEED')
plt.title('Boxplot of Bike Speed')
plt.show()

# Scatter plot for 'BIKE_COST' vs 'BIKE_SPEED'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_comma, x='BIKE_COST', y='BIKE_SPEED')
plt.title('Scatter Plot of Bike Cost vs Bike Speed')
plt.show()

# 5.2 Temporal Trends

# Line graph for trends over time (using 'OCC_YEAR' for example)
plt.figure(figsize=(10, 6))
data_comma['OCC_YEAR'].value_counts().sort_index().plot(kind='line')
plt.title('Bicycle Thefts Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Thefts')
plt.show()

# 5.3 Correlation Heatmaps
# Select only the numerical columns for correlation calculation
numerical_data = data_comma.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix for numerical data
corr_matrix = numerical_data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()



import geopandas as gpd
from shapely.geometry import Point

# Assuming 'LONG_WGS84' and 'LAT_WGS84' are the longitude and latitude columns in your dataset
# Convert the DataFrame to a GeoDataFrame
gdf = gpd.GeoDataFrame(
    data_comma, geometry=gpd.points_from_xy(data_comma.LONG_WGS84, data_comma.LAT_WGS84))

# Set the coordinate reference system (CRS) for the GeoDataFrame
gdf.set_crs(epsg=4326, inplace=True)

# Plotting the geospatial data
plt.figure(figsize=(15, 10))
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
base = world[world.name == 'Canada'].plot(color='white', edgecolor='black')
gdf.plot(ax=base, marker='o', color='red', markersize=5)
plt.title('Geospatial Visualization of Bike Thefts in Toronto')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()



import folium
from folium.plugins import HeatMap

# Assuming 'LAT_WGS84' and 'LONG_WGS84' are the latitude and longitude columns in your dataset
lat_lon_pairs = list(zip(data_comma.LAT_WGS84, data_comma.LONG_WGS84))

# Create a map centered around Toronto
map_toronto = folium.Map(location=[43.651070, -79.347015], zoom_start=12)

# Add a heatmap to the map
HeatMap(lat_lon_pairs).add_to(map_toronto)

# Display the map
map_toronto.save('C:/Users/parth/Downloads/Toronto_Bike_Thefts_Heatmap.html')





