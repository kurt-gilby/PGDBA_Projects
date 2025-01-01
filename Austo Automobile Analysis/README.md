# Austo Automobile Analysis

## **Project Overview**
This project analyzes customer purchase behavior and demographic data from Austo Motor Company, a leading car manufacturer specializing in SUVs, Sedans, and Hatchbacks. The primary focus is to derive actionable insights to optimize marketing campaigns, improve customer experience, and align product offerings with customer preferences. Additionally, the study evaluates the relationship between customer attributes and purchase patterns to recommend targeted marketing strategies.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Technologies Used](#technologies-used)
3. [Dataset Description](#dataset-description)
4. [Key Objectives](#key-objectives)
5. [Methodology](#methodology)
6. [Findings and Insights](#findings-and-insights)
7. [Recommendations](#recommendations)
8. [Acknowledgments](#acknowledgments)

---

## **Introduction**
The Austo Motor Company, during a board meeting, raised concerns regarding the efficiency of their current marketing campaign. The project aims to analyze customer data to answer key business questions related to gender, income, car type preferences, and pricing, while providing insights to enhance customer satisfaction and boost sales.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - **NumPy**: Numerical operations and array manipulation.
  - **Pandas**: Data manipulation and analysis.
  - **Matplotlib**:
    - **Matplotlib.pyplot**: Plotting and visualization.
    - **Matplotlib.ticker**: Customizing tick marks in visualizations.
  - **Seaborn**: Statistical data visualization.

---

## **Dataset Description**
The dataset contains approximately **15,000 customer purchase records**, with attributes spanning:
- **Demographic Data**: Gender, Age, Profession, Education, Number of Dependents, Marital Status
- **Financial Data**: Salary, Partner Salary, Total Salary, Loan Details
- **Automobile Data**: Price, Make (SUV, Sedan, Hatchback)

Key preprocessing steps included:
- Handling missing values.
- Correcting errors in Gender and Total Salary attributes.
- Converting numerical attributes (e.g., Dependents, Age) into categorical bins for analysis.

---

## **Key Objectives**
1. **Gender and Car Preferences**:
   - Analyze whether gender influences car type preferences.
2. **Income and Spending Behavior**:
   - Investigate how salary and financial status impact automobile purchases.
3. **Loan Influence**:
   - Examine how personal loans affect the amount spent on cars.
4. **Working Partner Impact**:
   - Assess whether having a working partner correlates with higher-priced car purchases.
5. **Marketing Campaign Optimization**:
   - Provide tailored recommendations for different customer segments.

---

## **Methodology**
1. **Exploratory Data Analysis (EDA)**:
   - Univariate and bivariate analysis.
   - Distribution visualization by gender, age, and salary categories.
2. **Statistical Analysis**:
   - Proportion tests for gender preferences.
   - Median and mean comparison for spending behavior.
3. **Segmentation Analysis**:
   - Grouping by attributes like gender, age, and loan status to identify patterns.
4. **Visualization**:
   - Used bar plots, box plots, and scatter plots to illustrate findings.

---

## **Findings and Insights**
1. **Gender and Car Preferences**:
   - Females prefer SUVs (53%), while males show a stronger preference for Hatchbacks (43%).
   - Females spend, on average, **₹15K more** on automobiles than males.

2. **Income and Spending Behavior**:
   - Salaried individuals are more likely to purchase Sedans (44%) compared to Hatchbacks and SUVs.
   - Individuals without loans show slightly higher spending capabilities.

3. **Loan Influence**:
   - No significant difference in spending between individuals with and without personal loans.

4. **Working Partner Impact**:
   - 14% of customers with a working partner purchased higher-priced cars compared to 16% without.

5. **Age-Specific Preferences**:
   - Younger individuals (22–25 years) prefer Hatchbacks and lower price points.
   - Middle-aged customers (30–45 years) lean towards Sedans in the mid-to-high price range.
   - Older customers (46+ years) predominantly purchase SUVs at high price points.

---

## **Recommendations**
1. **Gender-Targeted Campaigns**:
   - Market mid-to-high price SUVs to females, highlighting features they prefer.
   - Target males with low-range Hatchbacks and Sedans.

2. **Age-Targeted Campaigns**:
   - Design campaigns for:
     - Hatchbacks for younger customers (22–25 years).
     - Sedans for middle-aged customers (30–45 years).
     - SUVs for older customers (46+ years).

3. **Loan-Based Strategies**:
   - Encourage salaried individuals to explore loan-based financing options for mid-to-high priced cars.

4. **Feature-Based Promotions**:
   - Analyze car features by gender and income segments to design product-specific marketing campaigns.

5. **Working Partner Influence**:
   - Offer joint financing or partner-based discounts to increase sales among households with working partners.

---

## **Acknowledgments**
This project was conducted as part of the PGP-DSBA program. Special thanks to the Austo Motor Company for providing the dataset and guidance throughout the analysis.

---

