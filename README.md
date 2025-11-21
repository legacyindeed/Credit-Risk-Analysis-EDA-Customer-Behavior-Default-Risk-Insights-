Credit Risk EDA

This project analyzes a synthetic credit risk dataset to understand customer financial behavior and identify key drivers of loan default. The analysis explores income, utilization, delinquencies, spending patterns, and customer segments commonly used in fintech and credit underwriting.

Project Goals
Explore customer demographics and financial behavior
Understand which factors drive higher default risk
Compare risk across income, utilization, and spending segments
Identify differences between defaulters and non-defaulters
Build a foundation for future credit risk modeling

Repository Structure
.
├── credit_risk_analysis.py       # Main analysis script
├── plot/                    # Saved visualizations
│   ├── age_distribution.png
│   ├── income_distribution.png
│   ├── utilization_distribution.png
│   └── correlation_heatmap.png
└── README.md


Data Preparation
Loaded dataset and inspected schema
Checked for nulls and removed missing values
Removed duplicates
Summarized numeric + categorical variables



Exploratory Analysis (EDA)

Key visualizations include:
Age Distribution
Income Distribution
Utilization Rate Distribution
Correlation Heatmap

These plots give a high-level understanding of portfolio behavior


Segmentation & Default Analysis

Default rates were analyzed across:

Income Bins (quantiles)
Default decreases as income increases — classic risk pattern.

Utilization Bins
Customers above 80% utilization show the highest default rates.

Spend Segments & Education Levels
Segments show meaningful differences in income and risk behavior.

Income Segments (Low, Mid, Upper Mid, High)
Clear monotonic decline in default risk from Low → High.

Defaulters vs Non-Defaulters
A comparison of average behaviors shows that defaulters typically:
Earn less income
Have higher utilization
Have more delinquencies
Have more recent inquiries
These are standard predictors in credit risk modeling.



Key Insights Summary
Income and utilization are strong drivers of default risk.
Customers using >80% of their credit lines show ~20% default rates.
Delinquencies and credit inquiries sharply differentiate risky customers.
Spend and education segments help explain behavior and targeting opportunities.

Tools Used
- Python
- pandas
- matplotlib
- seaborn

Author
Ibrahim Shardow
