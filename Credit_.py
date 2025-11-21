import pandas as pd

full_data = pd.read_csv("C:/Users/ishardow/Documents/Python_Class/credit_risk_dataset.csv")
print(full_data)


full_data.describe(include='object')


#Groupby Education Level
full_data.groupby("education_level")["age"].agg(['count','mean','max','min'])


#GroupBy Spend Segment

full_data.groupby("spend_segment")["income"].agg(['count','max','min','mean'])
full_data.groupby("spend_segment")["age"].agg(['count','max','min','mean'])










#%%
import csv

with open("C:/Users/ishardow/Documents/Python_Class/credit_risk_dataset.csv","r") as credit_:
    system = csv.DictReader(credit_)
    
    for line in system:
        print (line)







#%%%

import numpy as np

# make results repeatable if you rerun
np.random.seed(42)

# define age ranges per segment
age_ranges = {
    "Travel-heavy": (30, 65),   # older, more likely to travel
    "Foodie":       (22, 40),   # younger
    "Family":       (30, 55),   # parents
    "Driver":       (40, 70),   # older / suburban
    "Balanced":     (18, 75),   # all ages
}

# overwrite the age column based on spend_segment
ages = []

for seg in full_data["spend_segment"]:
    low, high = age_ranges[seg]
    ages.append(np.random.randint(low, high + 1))

full_data["age"] = ages









#%%

import numpy as np

np.random.seed(42)

age_ranges_by_education = {
    "High School": (16, 18),
    "Bachelor":    (18, 23),
    "Master":      (23, 40),
    "PhD":         (25, 55)
}

age_ranges_by_spend = {
    "Travel-heavy": (30, 65),
    "Foodie":       (22, 40),
    "Family":       (30, 55),
    "Driver":       (40, 70),
    "Balanced":     (18, 75),
}

final_ages = []

for edu, seg in zip(full_data["education_level"], full_data["spend_segment"]):
    # choose the INTERSECTION (overlap) between both ranges
    edu_low, edu_high = age_ranges_by_education[edu]
    seg_low, seg_high = age_ranges_by_spend[seg]
    
    low = max(edu_low, seg_low)
    high = min(edu_high, seg_high)

    # Safety check — if ranges don't intersect, fallback to education
    if low >= high:
        low, high = edu_low, edu_high

    final_ages.append(np.random.randint(low, high + 1))

full_data["age"] = final_ages




#%%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.scatter(full_data["age"], full_data["income"], alpha=0.3)
plt.title("Age vs Income")
plt.xlabel("Age")
plt.ylabel("Income")
plt.show()


plt.hist(full_data["age"], bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()



plt.hist(full_data["utilization_rate"], bins=30)
plt.title("Utilization Rate Distribution")
plt.xlabel("Utilization")
plt.ylabel("Count")
plt.show()


plt.hist(full_data["delinquency_12m"], bins=20)
plt.title("Delinquencies (12m)")
plt.show()



full_data["education_level"].value_counts().plot(kind="bar")
plt.title("Education Level Counts")
plt.show()


full_data["spend_segment"].value_counts().plot(kind="bar")
plt.title("Spend Segment Distribution")
plt.show()



average_income = full_data.groupby("age")["income"].mean()
plt.plot(average_income ,linestyle = 'dashed')
plt.xlabel("Age")
plt.ylabel("Average Income")
plt.title("Trend over age")
plt.show()


full_data.describe( include= "object")
full_data.groupby("education_level")["income"].agg(['count','max','mean'])




#Format1
num_df = full_data.select_dtypes(include=['float64','int64'])
sns.heatmap(num_df.corr(), annot=False, cmap="coolwarm")

#Format2
num_df = full_data.select_dtypes(include=['int64','float64'])
corr = num_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm")
plt.show()


