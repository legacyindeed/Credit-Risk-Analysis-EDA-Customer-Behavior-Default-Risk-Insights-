#Section 1 - Load Data
import pandas as pd

full_data = pd.read_csv("credit_risk_dataset.csv")
print(full_data)
print(full_data.head())
print(full_data.shape)


#Section2 - Basci Exploration

#Data Types
full_data.info()

#Summary of Variables
full_data.describe(include='object')

#Checking for Nulls
full_data.isnull().sum()


#SECTION 3 - Clean Data

#Handle Missing Data
full_data = full_data.dropna()

#Remove Duplicates
full_data = full_data.drop_duplicates()


#SECTION 4 - Exploratory Analysis

import matplotlib as plt

#Age Distribution
full_data["age"].hist(bins = 30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Counts")
plt.show()

# Income distribution
full_data['income'].hist(bins=30)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.show()

# Utilization Distribution
full_data['utilization_rate'].hist(bins=30)
plt.title("Utilization Rate Distribution")
plt.xlabel("Utilization")
plt.ylabel("Count")
plt.show()

# Default rate by spend segment
print(full_data.groupby("spend_segment")["default_12m"].mean())

# Default rate by education level
print(full_data.groupby("education_level")["default_12m"].mean())

# Average income by segment
print(full_data.groupby("spend_segment")["income"].mean())

# Correlation Heatmap
import seaborn as sns
plt.figure(figsize=(14, 12))
sns.heatmap(full_data.corr(numeric_only=True), cmap="coolwarm",vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()


#Section5
