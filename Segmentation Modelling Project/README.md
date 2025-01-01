# Segmentation Modelling Project

## **Project Overview**
This project focuses on using unsupervised machine learning techniques, specifically clustering and Principal Component Analysis (PCA), to analyze and segment data. The project involves identifying homogeneous groups based on similarities in features and deriving actionable insights.

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
The project applies clustering techniques such as Hierarchical and K-Means Clustering to group data into meaningful segments. PCA is also used for dimensionality reduction to handle complex datasets with a large number of features. The project provides statistical insights and actionable recommendations to improve decision-making processes.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - NumPy, Pandas (Data manipulation and analysis)
  - Scikit-learn (Machine Learning)
  - Matplotlib, Seaborn (Visualization)
  - Statsmodels (Statistical computations)

---

## **Dataset Description**
The datasets analyzed in this project include:
1. **Ads Data**:
   - Contains 23066 observations and 19 features.
   - Features include:
     - **Numerical**: Ad Size, Impressions, Clicks, Spend, Revenue, etc.
     - **Categorical**: Timestamp, InventoryType, Ad Type, Platform, etc.
   - Derived metrics:
     - Click-Through Rate (CTR)
     - Cost Per Thousand Impressions (CPM)
     - Cost Per Click (CPC)

2. **Census Data**:
   - Contains demographic information with 61 features across 640 observations.
   - Includes state-level data and gender ratio insights.

---

## **Key Objectives**
1. Perform Exploratory Data Analysis (EDA).
2. Preprocess and clean data (e.g., handling missing values, scaling).
3. Apply clustering techniques:
   - Hierarchical Clustering.
   - K-Means Clustering.
4. Use PCA for dimensionality reduction and feature selection.
5. Provide actionable recommendations based on cluster profiles.

---

## **Methodology**
### **Step 1: Exploratory Data Analysis**
- Univariate and bivariate analysis.
- Identification of correlations and outliers.
- Scaling features for clustering.

### **Step 2: Clustering**
- **Hierarchical Clustering**:
  - Used Ward's linkage and Euclidean distance to create a dendrogram.
  - Determined optimal clusters using Silhouette Scores.

- **K-Means Clustering**:
  - Applied the Elbow Method and Silhouette Scores to select optimal `k`.
  - Examined cluster centroids for interpretability.

### **Step 3: Principal Component Analysis**
- Created covariance matrix.
- Determined optimal number of principal components to retain 90% explained variance.
- Analyzed eigenvalues and eigenvectors to understand feature contributions.

---

## **Findings and Insights**
### **Ads Data Clustering**
- Two distinct groups:
  - **Cluster 1**: Smaller ads with lower CPM and higher revenue.
  - **Cluster 2**: Larger ads with higher CPM and lower revenue.
- Recommended focusing on smaller ads for better ROI.

### **Census Data PCA**
- Reduced 61 features to 6 principal components while retaining 90% of the variance.
- Identified key contributing features for each principal component:
  - TOT_M, MARG_CL_M, MAIN_AL_F, etc.

---

## **Recommendations**
1. For Ads Data:
   - Prioritize small ads with dimensions around 72276 for higher revenue.
   - Focus on platforms and formats that maximize impressions and clicks.

2. For Census Data:
   - Use PCA-reduced dimensions for demographic insights and resource allocation.
   - Target underrepresented districts for policy intervention based on PCA clusters.

---

## **Acknowledgments**
This project was completed as part of the PGP-DSBA program requirements. Special thanks to the mentors and data providers for their support and guidance.

---

