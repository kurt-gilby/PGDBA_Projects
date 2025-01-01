# Project Title: Time Series Analysis and Forecasting for Wine Sales

## 📄 Project Description
This project focuses on analyzing and forecasting wine sales trends for **Rose Wine** and **Sparkling Wine** using historical data provided by **ABC Estate Wines**. The data spans monthly sales from **1980 to 1995**, and the objective is to derive actionable insights and create robust forecasting models to support strategic decision-making in sales optimization.

---

## 🛠️ Tools and Techniques Used
- **Programming Language:** Python
- **Libraries/Frameworks:**
  - Pandas, NumPy
  - Matplotlib, Seaborn
  - Statsmodels
- **Techniques:**
  - Data Cleaning and Preprocessing
  - Exploratory Data Analysis (EDA)
  - Decomposition of Time Series Data (Trend, Seasonality, Residuals)
  - Stationarity Testing (ADF Test)
  - Model Building and Evaluation:
    - Linear Regression
    - Simple Average
    - Moving Average
    - Exponential Smoothing (Single, Double, Triple)
    - ARIMA and SARIMA
  - Evaluation Metrics:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)

---

## ✨ Key Results and Insights
- **Rose Wine Sales**:
  - Exhibits a steady decline over the years with seasonality patterns.
  - Best performing model: **Double Exponential Smoothing** with the least MSE and MAE.
- **Sparkling Wine Sales**:
  - Sales are significantly higher than Rose and show seasonal trends.
  - Best performing model: **Triple Exponential Smoothing**, achieving the lowest MSE and MAE.
- **Stationarity**:
  - Both time series are non-stationary but achieve stationarity after differencing.
  - ADF test confirms stationarity post-transformation.
- **ARIMA and SARIMA**:
  - Auto ARIMA and SARIMA were explored for advanced modeling, but Triple Exponential Smoothing showed better performance for Sparkling Wine.

---

## 🚀 Instructions for Running the Code
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/project-repo-name.git
   cd project-repo-name
2. **Install the requirements.txt**
   ```bash
   pip install -r .\requirements.txt
   pip install openpyxl
3. The input data file is in ".\data"
4. Run the script file "example.py" in python environment, or better still run the jupyter notebook "example.ipynb", this file has detailed explanation of the approach taken for the project and the callouts.
5. The script creates an images folder to hold all the charts/plots.
6. **Issues Installing pmdarima module windows system**
As of creating this document pmdarima is not compatible with Python 3.13, this code has been created using Python 3.11.0 and numpy version 1.26.4
   
