# Predictive Modelling Project

## **Project Overview**
This project focuses on using predictive modeling techniques to solve real-world problems. It covers both regression and classification tasks using datasets that provide workstation activity data and demographic/socio-economic data. The analysis leverages supervised learning models such as Linear Regression, Decision Tree Regressor, Logistic Regression, and Decision Tree Classifier to derive actionable insights.

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
The project is part of the PGP-DSBA program coursework, focusing on supervised learning methods. It applies tools such as Linear Regression, Logistic Regression, and Decision Tree models to solve two distinct problems:
1. **Problem 1**: Predicting CPU usage on a workstation based on activity measures.
2. **Problem 2**: Predicting contraceptive use among married females using demographic and socio-economic data.

---

## **Technologies Used**
- Python
- Machine Learning Algorithms:
  - Linear Regression
  - Logistic Regression
  - Decision Tree (Classifier and Regressor)
  - Linear Discriminant Analysis (LDA)
- Data Preprocessing Techniques
- Visualization Libraries:
  - Matplotlib
  - Seaborn

---

## **Dataset Description**
### **Problem 1: Workstation Activity Data**
- **Purpose**: Predict CPU usage (`usr`) based on activity measures.
- **Features**: 21 independent features and 1 target feature (`usr`).
- **Key Attributes**:
  - `lread`, `lwrite`: Memory reads and writes per second.
  - `scall`, `sread`: System calls and reads per second.
  - `fork`, `exec`: Fork and exec calls per second.
  - `freemem`, `freeswap`: Free memory and swap space.

### **Problem 2: Demographic and Socio-Economic Survey Data**
- **Purpose**: Predict contraceptive use among married females.
- **Features**: 9 features and 1 target variable (`Contraceptive_method_used`).
- **Key Attributes**:
  - `Wife_age`, `No_of_children_born`: Numerical features.
  - `Wife_education`, `Husband_education`: Categorical education levels.
  - `Standard_of_living_index`, `Media_exposure`: Socio-economic indicators.

---

## **Key Objectives and Steps**
### **Problem 1: Workstation CPU Prediction**
1. Perform Exploratory Data Analysis (EDA).
2. Handle missing values and outliers.
3. Scale and transform features to address skewness.
4. Build and compare models:
   - Linear Regression
   - Decision Tree Regressor
5. Evaluate models using metrics:
   - Mean Squared Error (MSE)
   - R-squared (R²)
6. Derive insights and make recommendations.

### **Problem 2: Contraceptive Method Prediction**
1. Perform EDA on categorical and numerical features.
2. Encode categorical variables and scale numerical features.
3. Build and compare models:
   - Logistic Regression
   - Linear Discriminant Analysis (LDA)
   - Decision Tree Classifier
4. Evaluate models using:
   - Precision, Recall, F1-Score
   - ROC-AUC
5. Recommend the best model based on use-case requirements.

---

## **Model Development and Performance**
### **Problem 1: Regression Models**
- **Linear Regression**:
  - MSE: ~124 (Train), ~134 (Test)
  - R²: ~62.7% (Train), ~61.9% (Test)
- **Decision Tree Regressor**:
  - Overfitting observed, resolved through pruning.
  - Post-pruning, achieved competitive results.

### **Problem 2: Classification Models**
- **Logistic Regression**:
  - Strong recall but moderate precision.
- **LDA**:
  - Balanced performance but slightly lower than Decision Tree.
- **Decision Tree Classifier**:
  - Post-pruning achieved the highest F1-Score and ROC-AUC.

---

## **Insights and Recommendations**
### **Problem 1: CPU Usage Prediction**
1. **Linear Regression**:
   - Coefficients indicate `freeswap` has the highest positive impact, while `runqsz` has the most inverse impact.
   - Further feature engineering (e.g., PCA) is recommended to improve performance.
2. **Decision Tree Regressor**:
   - Provides better fit post-pruning but requires frequent retraining with live data.

### **Problem 2: Contraceptive Method Prediction**
1. **Logistic Regression**:
   - Needs additional feature engineering for better generalization.
2. **Decision Tree Classifier**:
   - Recommended model due to its superior performance on F1-Score and ROC-AUC.

---

## **How to Use**
1. **Clone the repository** and load the provided datasets.
2. **Run the Python scripts** for EDA, preprocessing, and model training.
3. Review the outputs for key metrics, insights, and recommendations.
4. Use the models for predictions or integrate into business workflows.

---

## **Acknowledgments**
This project was completed as part of the PGP-DSBA program. Special thanks to the course instructors and the data providers for their support.

---

