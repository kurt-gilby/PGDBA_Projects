# Project Title: Default Prediction & Stock Market Risk Analysis

## 📄 Project Description
The **Default Prediction & Stock Market Risk Analysis** focuses on building a Financial Health Assessment Tool for evaluating the financial stability and creditworthiness of companies. The tool uses machine learning to classify companies as defaulters or non-defaulters based on their financial metrics, enabling stakeholders to:
- Identify companies at financial risk.
- Make informed decisions regarding debt management and investments.
- Implement proactive risk mitigation strategies.

This project also includes a stock market analysis, highlighting stock performance trends, returns, and volatility for improved investment strategies.

---

## 🛠️ Tools and Techniques Used
- **Programming Language:** Python
- **Libraries/Frameworks:** Pandas, NumPy, Scikit-Learn, Matplotlib, SMOTE
- **Techniques:**
  - Data Cleaning and Preprocessing
  - Exploratory Data Analysis (EDA)
  - Feature Selection (Variance Inflation Factor, P-Value Analysis)
  - Machine Learning Models:
    - Logistic Regression
    - Random Forest
  - Model Optimization:
    - Hyperparameter Tuning
    - Synthetic Minority Oversampling Technique (SMOTE)
  - Evaluation Metrics:
    - Recall, Precision, ROC-AUC, Accuracy
  - Stock Analysis:
    - Time-Series Plotting
    - Return Calculations
    - Volatility Analysis (Mean vs. Standard Deviation)

---

## ✨ Key Results and Insights
### Part A: Financial Health Assessment
- **Logistic Regression**:
  - Initial recall for defaulters: 0.01 (poor performance).
  - Improved recall post-feature reduction and SMOTE: 0.25.
- **Random Forest**:
  - Recall on test set post-optimization: 0.22.
- Neither model performed adequately for real-world application. Recommendations for PCA or ensemble techniques like Boosting/Bagging for further improvement.

### Part B: Stock Market Analysis
- Stocks analyzed: Bharti Airtel, Yes Bank, ITC Limited, DLF, Tata Motors.
- **Insights**:
  - Bharti Airtel and ITC Limited provide mid-range returns with low volatility.
  - Yes Bank has high volatility with the least returns.
  - Diversification strategies combining low-risk and high-return stocks are optimal for risk-adjusted returns.

---

## 🚀 Instructions for Running the Code
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Default Prediction & Stock Market Risk Analysis.git
   cd Default Prediction & Stock Market Risk Analysis
