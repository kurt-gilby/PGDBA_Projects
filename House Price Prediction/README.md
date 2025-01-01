# House Price Prediction

## **Project Overview**
This project aims to accurately predict house prices based on various property features, such as location, size, and amenities. By leveraging data analytics and machine learning techniques, this project provides valuable insights for real estate businesses, sellers, and buyers to make informed decisions.

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
Real estate pricing involves complex factors influenced by structural, geographical, and temporal features. This project tackles the challenge of predicting house prices using a structured approach that incorporates exploratory data analysis (EDA), variable treatment, and advanced machine learning models.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - **NumPy**: For numerical computations.
  - **Pandas**: For data manipulation and preprocessing.
  - **Matplotlib** and **Seaborn**: For data visualization.
  - **Scikit-learn**: For machine learning model development.
  - **LightGBM** and **XGBoost**: For gradient boosting algorithms.
  - **Statsmodels**: For statistical analysis.

---

## **Dataset Description**
- **Size**: 21,613 rows and 23 columns.
- **Features**:
  - **Continuous Variables**: Living area, lot size, above ground area, and basement area.
  - **Categorical Variables**: Waterfront view, condition, construction grade, and furnishing status.
  - **Geographical Variables**: Latitude, longitude, and zip codes.
- **Target Variable**: Log-transformed house price for better prediction accuracy.

---

## **Key Objectives**
1. **Exploratory Data Analysis**:
   - Analyze the distribution of continuous and categorical variables.
   - Identify relationships between features and house prices.
   - Detect and treat missing values and outliers.
2. **Feature Engineering**:
   - Transform variables (e.g., log-transformation for skewed features).
   - Create new features like "house age" and "years since renovation."
3. **Model Building and Evaluation**:
   - Train and test various regression models to predict house prices.
   - Evaluate model performance using metrics like RMSE, MAPE, and R².

---

## **Methodology**
1. **Data Preprocessing**:
   - Imputation of missing values using KNN and median methods.
   - Removal of irrelevant columns and transformation of skewed variables.
2. **Exploratory Data Analysis**:
   - Univariate and bivariate analyses to derive insights from the data.
   - Correlation heatmaps and scatter plots to understand variable relationships.
3. **Clustering Analysis**:
   - Applied K-Means clustering to segment properties into 5 clusters based on features.
4. **Model Development**:
   - Supervised learning models were built using linear regression, ridge regression, decision trees, and ensemble models like LightGBM and XGBoost.
   - Hyperparameter tuning was performed to optimize model performance.

---

## **Findings and Insights**
1. **Key Influencers of House Prices**:
   - Larger living area and newer construction have a positive impact on prices.
   - Waterfront views and high construction grades significantly boost property value.
2. **Cluster Insights**:
   - Identified 5 property clusters, ranging from budget-friendly older homes to premium waterfront properties.
3. **Model Performance**:
   - **Best Model**: LightGBM Regressor.
   - **Metrics**:
     - RMSE (Test): 0.2077
     - MAPE (Test): 1.16%
     - R² (Test): 0.839
   - The LightGBM model demonstrated strong predictive accuracy and generalization.

---

## **Recommendations**
1. **Data Insights**:
   - Focus on features like living area, waterfront views, and construction grade to price properties effectively.
   - Use cluster analysis to target different market segments.
2. **Business Strategies**:
   - Offer personalized marketing for high-value properties.
   - Highlight premium features (e.g., waterfront view) in listings to attract buyers.
3. **Operational Improvements**:
   - Automate price predictions for real-time updates using the LightGBM model.

---

## **Acknowledgments**
This project was completed as part of the PGP-DSBA program. Special thanks to the mentors and peers for their support and guidance.

---
