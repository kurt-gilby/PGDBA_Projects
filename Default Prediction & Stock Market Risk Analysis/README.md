# Default Prediction & Stock Market Risk Analysis

## **Project Overview**
This project tackles two key financial problems:
1. **Default Prediction**: Develops a predictive model to classify companies as defaulters or non-defaulters based on financial metrics and net worth projections.
2. **Stock Market Risk Analysis**: Analyzes stock price trends, returns, and risk to provide actionable insights for investment decision-making.

The project leverages statistical methods, machine learning, and financial analytics to address business challenges and deliver actionable recommendations.

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
**Part A: Default Prediction**
The venture capitalists aim to build a Financial Health Assessment Tool to evaluate the financial stability of companies. The tool classifies companies as defaulters or non-defaulters based on whether their **Net Worth Next Year** is positive or negative, aiding debt management and credit risk evaluation.

**Part B: Stock Market Risk Analysis**
Analyzes stock price trends, returns, and volatilities for five stocks to identify investment opportunities based on risk-return profiles. The focus is on determining which stocks suit risk-averse and risk-tolerant investors.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - **Pandas**: For data manipulation and preprocessing.
  - **NumPy**: For numerical computations.
  - **Matplotlib** and **Seaborn**: For data visualization.
  - **Scikit-learn**: For machine learning models.
  - **Statsmodels**: For statistical analysis.

---

## **Dataset Description**
**Part A: Default Prediction**
- **Observations**: 4,256 rows.
- **Features**: 51 columns including financial metrics such as:
  - Total assets, total liabilities, equity, debt-to-equity ratio.
  - Revenue, profit margins, cash flow metrics, and liquidity ratios.
- **Target Variable**: "Net Worth Next Year" (Positive: Non-Defaulter, Negative: Defaulter).

**Part B: Stock Market Risk Analysis**
- **Stocks**: Bharti Airtel, Yes Bank, ITC Limited, DLF, and Tata.
- **Features**:
  - Weekly stock prices over a given period.
  - Stock returns calculated as percentage changes week-over-week.

---

## **Key Objectives**
### **Part A: Default Prediction**
1. Perform EDA to identify trends and relationships in financial metrics.
2. Handle missing values and outliers to prepare the dataset.
3. Build and evaluate predictive models (Logistic Regression, Random Forest).
4. Improve model performance using feature selection, hyperparameter tuning, and SMOTE.

### **Part B: Stock Market Risk Analysis**
1. Visualize stock price trends and identify growth or volatility patterns.
2. Calculate stock returns and evaluate mean and standard deviation.
3. Provide actionable insights for portfolio diversification and investment strategies.

---

## **Methodology**
### **Part A: Default Prediction**
1. **EDA**:
   - Univariate and multivariate analysis of financial metrics.
   - Heatmaps and pair plots to identify relationships between features.
2. **Data Preprocessing**:
   - KNN imputation for missing values.
   - Outlier treatment using interquartile range (IQR).
   - Feature scaling for model compatibility.
3. **Model Development**:
   - Built Logistic Regression and Random Forest models.
   - Evaluated models on metrics like recall, accuracy, and precision.
   - Improved performance using feature selection (VIF, p-values) and SMOTE.
4. **Performance Analysis**:
   - Compared model performance across multiple iterations.
   - Identified limitations and potential areas for improvement.

### **Part B: Stock Market Risk Analysis**
1. **Visualization**:
   - Plotted stock price trends over time.
   - Analyzed mean returns vs standard deviation (volatility).
2. **Statistical Analysis**:
   - Evaluated risk-return tradeoffs for each stock.
3. **Insights**:
   - Highlighted stocks suitable for risk-averse and risk-tolerant investors.

---

## **Findings and Insights**
### **Part A: Default Prediction**
1. **Model Performance**:
   - Logistic Regression achieved 25% recall after feature selection and SMOTE.
   - Random Forest achieved 22% recall, indicating room for model improvement.
2. **Key Predictors**:
   - Raw material turnover, total expenses, reserves and funds, current ratio, and cash profit as % of total income.
3. **Challenges**:
   - Both models underperformed, suggesting the need for alternative approaches like boosting or PCA.

### **Part B: Stock Market Risk Analysis**
1. **Stock Performance**:
   - Bharti Airtel and ITC Limited offered stable returns with low volatility.
   - DLF and Tata showed higher returns but moderate volatility.
   - Yes Bank exhibited the lowest returns and highest risk.
2. **Risk-Return Analysis**:
   - Bharti Airtel and ITC Limited are ideal for risk-averse investors.
   - DLF and Tata are suitable for risk-tolerant investors seeking higher returns.

---

## **Recommendations**
### **Part A: Default Prediction**
1. Explore advanced machine learning models like XGBoost or ensemble methods to improve recall.
2. Use dimensionality reduction (e.g., PCA) to address multicollinearity among features.
3. Increase dataset size or include additional relevant features for better model training.

### **Part B: Stock Market Risk Analysis**
1. For risk-averse investors, prioritize stocks like Bharti Airtel and ITC Limited.
2. Diversify portfolios by combining high-return stocks (DLF, Tata) with low-risk stocks.
3. Monitor weekly trends and adjust portfolios based on market performance.

---

## **Acknowledgments**
This project was completed as part of the PGP-DSBA program. Special thanks to mentors and team members for their valuable guidance and support.

---
