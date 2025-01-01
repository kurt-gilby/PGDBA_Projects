# Election Vote Prediction

## **Project Overview**
This project focuses on analyzing voter behavior and trends using a comprehensive survey dataset collected during elections. By leveraging machine learning models and ensemble techniques, the project aims to predict the political party a voter is likely to support. Additionally, it derives actionable insights from the dataset, such as the impact of demographic and socio-economic factors on voting preferences.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Technologies Used](#technologies-used)
3. [Dataset Description](#dataset-description)
4. [Key Objectives and Steps](#key-objectives-and-steps)
5. [Model Development and Performance](#model-development-and-performance)
6. [Insights and Recommendations](#insights-and-recommendations)
7. [How to Use](#how-to-use)
8. [Acknowledgments](#acknowledgments)

---

## **Introduction**
This project is a part of the **Machine Learning 2** coursework in the Postgraduate Program in Data Science and Business Analytics (PGP-DSBA). The study uses various machine learning models, including Naive Bayes, K-Nearest Neighbors (KNN), and ensemble methods (Bagging and Boosting), to predict voter preferences.

---

## **Technologies Used**
- Python (with libraries such as scikit-learn, pandas, numpy, matplotlib)
- Machine Learning Algorithms: Naive Bayes, KNN, Bagging, and Boosting
- Text Analytics (for secondary objectives)
- Data Preprocessing and Exploratory Data Analysis (EDA)

---

## **Dataset Description**
The dataset is derived from a survey conducted by CNBE, capturing perspectives from 1525 voters across multiple demographic and socio-economic factors. It contains 9 variables:
- **vote**: Political party preference (Labour or Conservative)
- **age**: Age of the voter
- **economic.cond.national**: National economic condition (1-5 scale)
- **economic.cond.household**: Household economic condition (1-5 scale)
- **Blair**: Assessment of the Labour leader (1-5 scale)
- **Hague**: Assessment of the Conservative leader (1-5 scale)
- **Europe**: Euroscepticism score (0-10 scale)
- **political.knowledge**: Knowledge of party positions on European integration (0-3 scale)
- **gender**: Gender of the voter (Female or Male)

---

## **Key Objectives and Steps**
1. Perform **Exploratory Data Analysis**:
   - Analyze distributions, identify patterns, and check data integrity.
2. Conduct **Data Preprocessing**:
   - Handle duplicates, encode categorical variables, and scale features.
3. Build Machine Learning Models:
   - Implement Naive Bayes, KNN, Bagging, and Boosting classifiers.
4. Evaluate Model Performance:
   - Use metrics like ROC-AUC, precision, recall, and F1-score for comparison.
5. Hyperparameter Tuning:
   - Optimize Bagging and Boosting models using GridSearchCV.
6. Select Final Model:
   - Based on performance metrics and interpretability.
7. Derive Insights:
   - Identify key factors influencing voter behavior.

---

## **Model Development and Performance**
### **Machine Learning Models Used**
1. **Naive Bayes**:
   - Baseline model with moderate performance.
2. **K-Nearest Neighbors (KNN)**:
   - Improved accuracy over Naive Bayes but less robust compared to ensemble methods.
3. **Bagging Classifier**:
   - Strong performance but showed signs of overfitting.
4. **Boosting Classifiers**:
   - Implemented AdaBoost and Gradient Boosting.
   - AdaBoost showed balanced performance with high precision and recall.

### **Performance Metrics**
- **Best Model**: AdaBoost Classifier
  - ROC-AUC Score: 0.89
  - Balanced precision and recall for both classes.
  - Robust generalization on test data.
- **Key Features for Prediction**:
  - Top factors influencing voter preferences:
    - **Blair** (Labour leader popularity)
    - **Europe** (Euroscepticism)
    - **Hague** (Conservative leader popularity)
    - **Age** (specific age groups skew voting trends)
    - **Economic conditions** (national perception)

---

## **Insights and Recommendations**
1. **Predictive Accuracy**:
   - The final model achieves ~80% accuracy in predicting voter preferences.
2. **Key Determinants**:
   - Leadership ratings, Euroscepticism, and age are strong indicators of voting behavior.
3. **Future Work**:
   - Explore advanced models like XGBoost and LightGBM.
   - Conduct deeper analysis on interactions between demographic and socio-economic factors.

---

## **How to Use**
1. Clone the repository and load the dataset into a Python environment.
2. Run the provided Python scripts for data preprocessing, model training, and evaluation.
3. Review the outputs, including confusion matrices, classification reports, and feature importance plots.

---

## **Acknowledgments**
This project was conducted as part of the PGP-DSBA coursework. Special thanks to **CNBE News Channel** for providing the survey dataset.

---

