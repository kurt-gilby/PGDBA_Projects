# Time Series Analysis and Forecasting for Wine Sales

## **Project Overview**
This project analyzes historical wine sales data for **Rose** and **Sparkling Wine** varieties from 1980 to 1995. Using time series forecasting techniques, the project identifies trends, seasonality, and key patterns in the sales data. The ultimate goal is to develop accurate forecasting models to assist ABC Estate Wines in strategic decision-making and optimizing sales performance.

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
The dataset comprises monthly sales data for Rose and Sparkling Wine varieties, representing their historical performance over 15 years. The project explores trends, seasonality, and stationarity in the data while building forecasting models such as ARIMA, SARIMA, and Exponential Smoothing. These models aim to predict future sales and provide actionable insights for decision-makers.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - **NumPy**: For numerical operations.
  - **Pandas**: For data manipulation and preprocessing.
  - **Matplotlib** and **Seaborn**: For data visualization.
  - **Statsmodels**: For time series decomposition and ARIMA modeling.
  - **pmdarima**: For Auto ARIMA modeling.

---

## **Dataset Description**
- **Source**: ABC Estate Wines
- **Duration**: 1980–1995
- **Frequency**: Monthly
- **Attributes**:
  - **Rose Wine Sales**: Sales in units with some missing values.
  - **Sparkling Wine Sales**: Higher sales volume with no missing values.

---

## **Key Objectives**
1. Perform **Exploratory Data Analysis (EDA)** to understand sales trends, seasonality, and outliers.
2. Preprocess the data to handle missing values and ensure stationarity.
3. Build forecasting models for sales prediction:
   - Linear Regression
   - Moving Averages
   - Exponential Smoothing
   - ARIMA and SARIMA models
4. Compare model performances using metrics like MSE and MAE.

---

## **Methodology**
1. **Exploratory Data Analysis**:
   - Analyzed trends and seasonality using line plots and rolling statistics.
   - Identified a declining trend in Rose sales and consistent seasonality in Sparkling sales.

2. **Data Preprocessing**:
   - Handled missing values in Rose sales using forward-fill (ffill) imputation.
   - Conducted stationarity tests (ADF Test) to assess the need for differencing.

3. **Model Development**:
   - Built models on original and stationary datasets.
   - Generated ACF and PACF plots to determine AR and MA orders for ARIMA models.
   - Implemented Auto ARIMA for automated parameter selection.
   - Evaluated models using train-test splits (70%-30%).

4. **Model Evaluation**:
   - Compared model performances using metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE).

---

## **Findings and Insights**
1. **Trends and Seasonality**:
   - **Rose Wine**: A declining trend in sales, with seasonality patterns observed in July–December.
   - **Sparkling Wine**: High and stable sales with significant seasonal spikes in March–April and July–December.

2. **Stationarity**:
   - Both datasets were non-stationary initially but achieved stationarity after differencing.

3. **Model Performance**:
   - **Rose Wine**: Double Exponential Smoothing performed best.
     - MSE: 216,753
     - MAE: 896.38
   - **Sparkling Wine**: Triple Exponential Smoothing outperformed other models.
     - MSE: 134,586
     - MAE: 292.08

---

## **Recommendations**
1. **Forecasting Models**:
   - Use Double Exponential Smoothing for Rose sales and Triple Exponential Smoothing for Sparkling sales to ensure accuracy in future forecasts.

2. **Marketing Strategies**:
   - Leverage seasonality insights to boost sales during peak periods (July–December for both wines).
   - Address the declining trend in Rose sales through targeted promotions and improved distribution.

3. **Operational Adjustments**:
   - Plan production and inventory levels based on forecasted demand.
   - Explore diversification strategies for Rose Wine to counter the sales decline.

---

## **Acknowledgments**
This project was conducted as part of the PGP-DSBA program. Special thanks to mentors and ABC Estate Wines for providing the dataset and guidance.

---
