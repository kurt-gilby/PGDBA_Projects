# Austo Automobile and GODIGT Bank Analysis

## **Project Overview**
This project involves two distinct but interrelated analyses:
1. **Austo Automobile Analysis**: Analyzes customer purchase behavior and demographic data from Austo Motor Company to optimize marketing campaigns and product offerings.
2. **GODIGT Bank Analysis**: Investigates customer credit card attrition and usage patterns to recommend strategies for reducing attrition and enhancing profitability.

The project leverages statistical and data analysis techniques to derive actionable insights and provide data-driven recommendations.

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
This project addresses two business problems:
1. **Austo Motor Company**: Concerns about inefficient marketing campaigns led to an analysis of customer preferences, spending patterns, and regional sales.
2. **GODIGT Bank**: High credit card attrition prompted an investigation into customer behaviors, card usage, and factors influencing profitability.

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
### **Austo Motor Company Data**:
- **Observations**: 15,000 customer records.
- **Attributes**:
  - **Demographics**: Gender, Age, Profession, Education, Dependents.
  - **Financial Data**: Salary, Partner Salary, Loan Details.
  - **Automobile Details**: Price, Make (SUV, Sedan, Hatchback).

### **GODIGT Bank Data**:
- **Observations**: 8,400 customer records.
- **Attributes**:
  - **Account Information**: Net worth, average spending (last 3 months), credit card limit.
  - **Card Usage**: Card type, activity status, and overall future month activity.

---

## **Key Objectives**
### **Austo Motor Company**:
1. Understand gender and age preferences for car types and price ranges.
2. Investigate the impact of loans and partner income on spending.
3. Analyze regional sales trends to identify high-performing areas.

### **GODIGT Bank**:
1. Identify factors influencing high credit card attrition.
2. Evaluate the relationship between net worth, spending, and credit card limits.
3. Recommend strategies to align customers with appropriate credit cards.

---

## **Methodology**
1. **Exploratory Data Analysis (EDA)**:
   - Performed on both datasets to identify patterns and trends.
   - Used bar plots, box plots, and heatmaps for visualization.

2. **Statistical Analysis**:
   - Conducted proportion tests and descriptive statistics.
   - Used clustering techniques for customer segmentation.

3. **Feature Importance Analysis** (GODIGT Bank):
   - Identified top features influencing credit card usage and attrition.

---

## **Findings and Insights**
### **Austo Motor Company**:
1. **Gender Preferences**:
   - Females prefer SUVs (53%), while males lean towards Hatchbacks (43%).
   - On average, females spend ₹15K more than males.

2. **Age-Specific Trends**:
   - Younger customers prefer Hatchbacks; middle-aged and older customers prefer Sedans and SUVs.
   - High-income older customers predominantly purchase SUVs.

3. **Regional Sales**:
   - District X and Region Y are top-performing areas, while District Z shows low sales.

4. **Loan and Partner Impact**:
   - Having a working partner has minimal impact on higher-priced car purchases.

### **GODIGT Bank**:
1. **Key Variables**:
   - **Net worth**, **average spending (last 3 months)**, and **credit card limit** are the top predictors of customer profitability.
2. **Credit Card Attrition**:
   - Customers with lower average spends and inactive credit cards have a higher likelihood of attrition.
3. **Card Type Preferences**:
   - High-net-worth individuals tend to prefer cards with higher limits and exclusive benefits.

---

## **Recommendations**
### **Austo Motor Company**:
1. Design gender-targeted campaigns:
   - Promote SUVs to females and Hatchbacks to males.
2. Regional campaigns:
   - Increase marketing budgets for high-performing regions.
   - Focus on improving sales in low-performing areas.
3. Feature-based promotions:
   - Highlight features aligned with gender-specific preferences.

### **GODIGT Bank**:
1. Align credit card offerings with customer profiles:
   - Offer high-limit cards to high-net-worth individuals.
   - Recommend budget-friendly cards to low-net-worth customers.
2. Enhance customer engagement:
   - Target inactive customers with special offers to reduce attrition.
3. Monitor spending trends:
   - Use customer spending patterns to design personalized rewards programs.

---

## **Acknowledgments**
This project was conducted as part of the PGP-DSBA program. Special thanks to Austo Motor Company and GODIGT Bank for providing datasets and to mentors for their guidance throughout the analysis.

---
