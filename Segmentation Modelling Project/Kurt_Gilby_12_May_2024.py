#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score as ss
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# # Machine Learning - 1

# ## Introduction
# While doing the Machine Learning -1 course, we have seen and practised some tools and techniques that fall under the  **unsupervised** learning criteria, which means that we look at all the independent features, and try and find some meaning, we do this by generating and selecting the most important features which explain most of the **variance** in the data ***"PCA"***, Or/And find similar observations/records and group them together, label them based on their similarities by the use of ***"Clustering"*** techniques.
# 
# I will in the following pages use the said tools and techniques to review the problem sets given and answer the questions posed. In the following pages of this document, I have given an **Overview, Dataset description, Objective and Information given** section, which gives added information where needed, the **questions asked** section under each, lists out the specific questions ask for each dataset/sample.
# 
# The first problem set here showcases two Clustering techniques ***"Hierarchical Clustering"*** and ***"K-means Clustering"***, while the secound problem set looks at the dataset given and showcases the **Principal Component Analysis** or ***"PCA"*** technique.
# 
# Along with using and showcasing the above I will also showcase statistical techniques and good practices learnt in pervious courses, such as **Exploratory Data Analysis (EDA)**, and **Data Preprocessing**.

# ## Problem 1 - Clustering

# ### Overview
# A Digital Marketing company **"ads24x7"** would like to segment ads into homogeneous groups, Groups having similar features.
# To do this I will use the data provided by the **"Marketing Intelligence"** team of the company, and use **Clustering** procedures to create the groups and try and find actionable insights and recommendations for the input provided.

# ### Dataset
# This is the Definition of the data provided by the "Marketing Intelligence" team.
# The features commonly used in digital marketing are three, namely **CTR,CPM and CPC** the definition is in the below table.
# | Sl. No | Column Name | Column Description |
# | --- | --- | --- |
# | 1 | Timestamp | The Timestamp of the particular Advertisement. |
# | 2 | InventoryType | The Inventory Type of the particular Advertisement. Format 1 to 7. This is a Categorical Variable. |
# | 3 | Ad - Length | The Length Dimension of the particular Adverstisement. |
# | 4 | Ad- Width | The Width Dimension of the particular Advertisement. |
# | 5 | Ad Size | The Overall Size of the particular Advertisement. Length*Width. |
# | 6 | Ad Type | The type of the particular Advertisement. This is a Categorical Variable. |
# | 7 | Platform | The platform in which the particular Advertisement is displayed. Web, Video or App. This is a Categorical Variable. |
# | 8 | Device Type | The type of the device which supports the partciular Advertisement. This is a Categorical Variable. |
# | 9 | Format | The Format in which the Advertisement is displayed. This is a Categorical Variable. |
# | 10 | Available_Impressions | How often the particular Advertisement is shown. An impression is counted each time an Advertisement is shown on a search result page or other site on a Network. |
# | 11 | Matched_Queries | Matched search queries data is pulled from Advertising Platform and consists of the exact searches typed into the search Engine that generated clicks for the particular Advertisement.   |
# | 12 | Impressions | The impression count of the particular Advertisement out of the total available impressions.  |
# | 13 | Clicks | It is a marketing metric that counts the number of times users have clicked on the particular advertisement to reach an online property. |
# | 14 | Spend | It is the amount of money spent on specific ad variations within a specific campaign or ad set. This metric helps regulate ad performance. |
# | 15 | Fee | The percentage of the Advertising Fees payable by Franchise Entities.  |
# | 16 | Revenue | It is the income that has been earned from the particular advertisement. |
# | 17 | CTR | CTR stands for "Click through rate". CTR is the number of clicks that your ad receives divided by the number of times your ad is shown. Formula used here is CTR = Total Measured Clicks / Total Measured Ad Impressions x 100. Note that the Total Measured Clicks refers to the 'Clicks' Column and the Total Measured Ad Impressions refers to the 'Impressions' Column. |
# | 18 | CPM | CPM stands for "cost per 1000 impressions." Formula used here is CPM = (Total Campaign Spend / Number of Impressions) * 1,000. Note that the Total Campaign Spend refers to the 'Spend' Column and the Number of Impressions refers to the 'Impressions' Column. |
# | 19 | CPC | CPC stands for "Cost-per-click". Cost-per-click (CPC) bidding means that you pay for each click on your ads. The Formula used here is CPC = Total Cost (spend) / Number of Clicks. Note that the Total Cost (spend) refers to the 'Spend' Column and the Number of Clicks refers to the 'Clicks' Column.|
# 

# ### Objective
# Using the dataset provided, I will do the following:
# 1. Define the Problem
# 2. Explore the Data
#     1. Check the shape of the data.
#     2. Check the data types
# 3. Get the statistical summary of the data.
#     1. Univariate Analysis
#     2. Bivariate Analysis
# 4. Perform Data Preprocessing
#     1. Check Duplicates
#     2. Check Missing Values
#     3. Treat Missing Values.
#     4. Check for Outliers
#     5. Treat Outliers
# 5. Perform Hierarchical Clustering
#     1. Construct a dendogram
#     2. Identify optimum number of Clusters
# 6. Perform K-means Clustering
#     1. Plot the Elbow Curve
#     2. Check Silhouette Scores
#     3. Figure out the appropriate number of Clusters
# 7. Do Cluster Profiling
# 8. Derive Actionable Insights and Recommendations

# ### Questions Asked

# #### Define the problem and perform Exploratory Data Analysis
# ##### Problem Definition
# Using the dataset given for ads campaigns run we need to find campaigns that are similar to one and other and derive actionable Insights and recommendations, for the said groups of campaigns.

# ##### Check the Data and do EDA
# 1. Check the Shape, Data Types.
# 2. Do Exploratory Data Analysis: Univariate and Bivariate.
#     1. Since we are doing Clustering, I will **Ignore the Categorical features for now, and do the Analysis only on the Numerical features, Ideally these can be treated too by using one hot key encoding or KModes kind of methods, but for this exercise I will concentrate only on the numerical variables for clustering.**

# ##### Key Observations & Teatments
# 1. The Ads Dataset has **23066** Observations and  **19** features.
# 2. The Ads Dataset has **6** features of datatype **string**, and there are:
#     1. **['Timestamp' 'InventoryType' 'Ad Type' 'Platform' 'Device Type' 'Format']**
# 3. The Ads Dataset has **13** features of datatype **number** , and there are:
#     1. **['Ad - Length' 'Ad- Width' 'Ad Size' 'Available_Impressions'
#      'Matched_Queries' 'Impressions' 'Clicks' 'Spend' 'Fee' 'Revenue' 'CTR'
#      'CPM' 'CPC']**
# 4. The datatype for **Timestamp** needs to be changed to datetime and used as a categorical feature.
# 5. The datatype for **InventoryType,Plaform,Device Type,Format** nedds to be changed to category.
# 6. There are Null/Missing values in **CTR,CPM and CPC** this needs to be treated.
# 7. **Since this data would be used for clustering, Ignore the Categorical features for now, and do the Analysis only on the Numerical features, Ideally these can be treated too by using one hot key encoding or KModes kind of methods, but for this exercise I will concentrate only on the numerical variables for clustering.**
# 8. The describe of the numerical features shows us:
#     1. There are missing values for CTR, CPM and CPC, as the count using the describe of the data is less than the 23066.
#     2. The scale of the values in most of the features are different form each other and scaling would need to be done.
#     3. There is potientially outliers persent, as for some features the max values is far from the 75 Percentile and median.
# 10. The Univariate Analysis on numerical features shows us:
#     1. We perform univariate analysis on all the numerical freatures by doing a boxplot on each.
#         1. Findings Ad - Length and Ad - Width does not have ouliers, The rest ['Ad Size' 'Available_Impressions' 'Matched_Queries' 'Impressions' 'Clicks' 'Spend' 'Fee' 'Revenue' 'CTR' 'CPM' 'CPC'] have outliers.
#         2. **Even though outliers are available we may want to keep them as removing the same would effect the classifiction in clustering.**
#     2. We perform univariate analysis on all the numerical freatures by doing a histograms on each.
#         1. Ad - Lenght, Ad - Width, Ad - Size and Fee looks to have few distcreet values.
#         2. The rest of the features expect the above and CTR, CPM and CPC follow a bell shape distribution with all being right skewed.
#         3. CTR, CPM and CPC looks to be bi-modal and rigth skewed.
#     3. We check the skewness of the data and see that.
#         1. ['Ad Size' 'Available_Impressions' 'Matched_Queries' 'Impressions' 'Clicks' 'Spend' 'Revenue' 'CPC'] are highly positively skewed
#         2. Fee is highly negatively skewed
#         3. ***Ideally we would go back to the raw dataset and do transformations like, square root, cube root, log or box-cox try and make the data less skewed, but for this exercise we will not do this***
# 11. We perform Bivariate analysis.
#     1. do a pairplot
#        1. Available_impressions seems to be correlated with Matched_Queries, Impressions, Clicks, Spend, Revenue.
#        2. Matched_Queries seems to be correlated with Available_impressions, Impressions, Clicks, Spend, Revenue
#        3. Impressions seems to correlated with  Available_impressions, Matched_Queries, Clicks, Spend, Revenue
#        4. Clicks seems to correlated with  Available_impressions, Matched_Queries, Impressions, Spend, Revenue
#        5. Spend seems to correlated with  Available_impressions, Matched_Queries, Impressions, Clicks, Revenue
#        6. Revenue seems to correlated with  Available_impressions, Matched_Queries, Impressions, Clicks, Spend
#        7. Fee seems to have a negative correlation with alll the above mentioned attributes.
#     2. do a heatmap
#        1. The heat map shows the same inferences as the pair plot.
#        2. Used "annot" but the numbers are not displayed but the color map confrims that there is alot of correlation between the above columns.
#     3. Since there is alot of correlated columns we will review and remove some of the ones before we do the clustering activities.

# In[2]:


# importing the dataset
ads_data = pd.read_excel("./data/Clustering_Clean_Ads_Data.xlsx")


# In[3]:


# Check the frist five observations
ads_data.head()


# In[4]:


#Get the number of observations and features
print(f"The Ads Dataset has {ads_data.shape[0]} rows and  {ads_data.shape[1]} columns") 


# In[5]:


#Use info the get high level summary of the dataset.
ads_data.info()

#Get the columns of type object and number.
obj_cols = ads_data.select_dtypes(include=['object']).columns
obj_cols
print(f"The Ads Dataset has {len(obj_cols)} string columns, and there are:")
print(f"{obj_cols.values}")
num_cols = ads_data.select_dtypes(include=['int','float']).columns
print(f"The Ads Dataset has {len(num_cols)} numeric columns, and there are:")
print(f"{num_cols.values}")


# In[6]:


#Create a Copy of the orignal dataset
ads_data_copy = ads_data.copy()


# In[7]:


#Change the dtypes for Timestamp, InventoryType, Ad Type, Platform, Device Type and Format
ads_data.info()
ads_data['Timestamp'] = pd.to_datetime(ads_data.Timestamp)
ads_data['Timestamp'] = ads_data['Timestamp'].astype('category')
ads_data['InventoryType'] = ads_data['InventoryType'].astype('category')
ads_data['Ad Type'] = ads_data['Ad Type'].astype('category')
ads_data['Platform'] = ads_data['Platform'].astype('category')
ads_data['Device Type'] = ads_data['Device Type'].astype('category')
ads_data['Format'] = ads_data['Format'].astype('category')
ads_data.info()
ads_data.head()


# In[8]:


obj_cols = ads_data.select_dtypes(exclude=['datetime','int','float']).columns
obj_cols
num_cols = ads_data.select_dtypes(include=['datetime','int','float']).columns
num_cols
ads_data[obj_cols].describe().T


# In[9]:


ads_data[num_cols].describe().round(2).T


# In[10]:


#univariate analysis for numerical columns:
# box plots
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(12, 9))  # Adjust figsize as needed

for i, col in enumerate(num_cols[:13]):
    if i < 5 :
        img_row = 0
        img_col = i
    elif i < 10:
        img_row = 1
        img_col = i-5
    else:
        img_row = 2
        img_col = i-10
    sns.boxplot(data=ads_data[col], ax=axes[img_row, img_col])  # Specify the correct axes
    axes[img_row, img_col].set_xticks([]) 
    axes[img_row, img_col].set_xlabel(col) 

fig.delaxes(axes[2][3])
fig.delaxes(axes[2][4])
fig.suptitle('Univariate Analysis of features using Box-Plots')
plt.draw()
plt.tight_layout()
plt.show()

fig.savefig('./images/uni_box_plot.svg')


# In[11]:


#univariate analysis for numerical columns:
# histograms
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(12, 9))  # Adjust figsize as needed

for i, col in enumerate(num_cols[:13]):
    if i < 5 :
        img_row = 0
        img_col = i
    elif i < 10:
        img_row = 1
        img_col = i-5
    else:
        img_row = 2
        img_col = i-10
    sns.histplot(data=ads_data[col],kde=True, ax=axes[img_row, img_col])  # Specify the correct axes
    
fig.delaxes(axes[2][3])
fig.delaxes(axes[2][4])
fig.suptitle('Univariate Analysis of features using Histograms')
plt.draw()
plt.tight_layout()
plt.show()

fig.savefig('./images/uni_hist_plot.svg')


# In[12]:


#Check the Skewness in the Data
skew = {}
for col in num_cols:
    skew[col] = ads_data[col].skew()
ads_skew = pd.DataFrame(skew, index=[0])
ads_skew = ads_skew.T.rename(columns={0: 'skew'})
sns.barplot(data=ads_skew,y='skew',x=ads_skew.index.values)
plt.xticks(rotation=90)
plt.title('Skew of features')
plt.xlabel('Features')
plt.tight_layout()
plt.savefig('./images/uni_skew_plot.svg')
plt.show()


# In[13]:


# do a pairplot on all the numerical columns
ads_num_data = ads_data[num_cols]
sns.pairplot(data=ads_num_data)
plt.savefig('./images/bi_pair_plot.jpg')
plt.show()


# In[14]:


ads_num_corr = ads_data[num_cols].corr().round(2)
matrix = np.triu(ads_num_corr)
annot = True
plt.figure(figsize=(12,10))
sns.heatmap(ads_num_corr, mask=matrix,annot=annot,fmt=".1f")#annot is not given the full values will check this later for now will go with the color heat map
plt.tight_layout()
plt.savefig('./images/bi_heat_plot.svg')
plt.show()


# #### Data Preprocessing
# We will focus our data prepossing steps only on the nummerical columns as, we will be using only these in the clustering exercise.
# ##### Missing Value Check and Treatment
# **CTR, CPM and CPC** have missing values and need to be treated, we will fill these values using the definitions given for each:
# 
# CTR = Total Measured Clicks / Total Measured Ad Impressions x 100 
# ***"Note that the Total Measured Clicks refers to the 'Clicks' Column in the dataset and the Total Measured Ad Impressions refers to the 'Impressions' Column in the dataset."***
# 
# CPM = (Total Campaign Spend / Number of Impressions) * 1,000 
# ***"Note that the Total Campaign Spend refers to the 'Spend' Column in the dataset and the Number of Impressions refers to the 'Impressions' Column in the dataset."***
# 
# CPC = Total Cost (spend) / Number of Clicks 
# ***"Note that the Total Cost (spend) refers to the 'Spend' Column in the dataset and the Number of Clicks refers to the 'Clicks' Column in the dataset."***
# ##### Outlier Treatment 
# There are Outliers in the dataset for 'Ad Size' 'Available_Impressions' 'Matched_Queries' 'Impressions' 'Clicks' 'Spend' 'Fee' 'Revenue' 'CTR' 'CPM' 'CPC'
# , But since we are doing Clustering and outlier may form a signification cluster I will keep the outliers with out treatment
# ##### Scaling
# There are features with different scales so we scale the data using the zscore method from the scipy libary.
# ##### Identification of Features to use in Clustering.
# 1. We will not use any of the categorical features: exculde : Timestamp, InventoryType, Ad Type, Platform, Device Type, and Format. Reason: Categorical Columns will need seprate treatment like one hot key encoding or KModes for example, which we will not explore in this exercise.
# 2. Ad-Size is made up as a calculated field of Ad - Length and Ad- Width: exclude : Ad- Length, Ad- Width. Reason: Both Length and Width is caputure in size.
# 3. Avalable_Impressions and Matched_Queries are highly correlated "0.99": exclude : Matched_Queries  Reason: Matched_Queries is not something the Ad agency can directly control.
# 4. Avalable_Impressions and Impressions are highly correlated "0.99": exclude : Avalable_Impressions  Reason: Impression is a better gauge of how a particular Advertisement is doing, as compared to the overall of avialble impressions.
# 5. Spend and Revenue are highly correlated "1": But we will keep both to just in case we have less spend high revenue clusters.
# 6. Final Columns used for Clustering: Ad Size, Impressions, Clicks, Spend, Fee, Revenue, CTR, CPM, CPC
# 
# 
# 
# 

# In[15]:


#check for duplicates
ads_data.duplicated().sum()


# In[16]:


#check for missing values
ads_data.isna().sum()


# In[17]:


#update the values for CTR, CPM and CPC
ads_data.CTR = (ads_data.Clicks/ads_data.Impressions)*100
ads_data.CPM = (ads_data.Spend/ads_data.Impressions)*1000
ads_data.CPC = (ads_data.Spend/ads_data.Clicks)*1000
#check for missing values
ads_data.isna().sum()


# In[18]:


#Scaling all the numercial columns to the same scale using zscore scaling
ads_num_data = ads_data[num_cols]
ads_num_data.head()
ads_scaled_data =zscore(ads_num_data)
ads_scaled_data.head()


# In[19]:


# Choose and create the final dataset for clustering analysis.
cols = ['Ad Size', 'Impressions', 'Clicks', 'Spend', 'Fee', 'Revenue', 'CTR', 'CPM', 'CPC']
ads_cluster_data = ads_scaled_data[cols]
ads_cluster_data


# #### Hierarchical Clustering
# ##### Construct a dendogram using Ward linkage and Euclidean distance
# We constructed the dendogram, using the Ward linkage.
# ##### Indentify optimal number of Clusters.
# Looking at the dendogarm it looks like 3 or 4 clusters should be optimal, but instead of relying on this we will calculate the Silhouette Score at various values of clusters to check what would be optimal.
# Doing this we see that it suggests that 2 or 5 clusters would be optimal with 2 having more clear boundaries between the clusters than 5, but 5 is also slightly higher than .5 and closer to 1 so would be good to investigate

# In[20]:


# Get the clusters and dendrogram based on Ward linkage
wardlink = linkage(ads_cluster_data,method= 'ward', metric='euclidean')


# In[21]:


plt.figure(figsize=(12,9))
dend = dendrogram(wardlink)
plt.ylabel('Euclidean Distance')
plt.xlabel('Samples or Observations')
plt.title('Dendogram Ads Data, Using Ward Linkage and Euclidean Distance')
plt.savefig('./images/scaled_dend_plot.svg')
plt.show()


# In[22]:


plt.figure(figsize=(12,9))
dend_p = dendrogram(wardlink, truncate_mode= 'lastp', p = 10)
plt.ylabel('Euclidean Distance')
plt.xlabel('Samples or Observations')
plt.title('Dendogram Ads Data Tuncate Mode P=10, Using Ward Linkage and Euclidean Distance')
plt.savefig('./images/scaled_dend_last_p_plot.svg')
plt.show()


# In[23]:


# finding the differrent siloutte scorce at different value of k and plotting the same
ss_scores = []
for k in range(2,20):
    labels = fcluster(wardlink,t=k, criterion='maxclust')
    sil = ss(ads_cluster_data,labels)
    ss_scores.append(sil)
    print(f"Silhouette Score (K={k}): {sil:.4f}")


# In[24]:


# Plotting Silhouette Scores to check which one would be optimal
# Plot the silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(range(2,20),ss_scores, marker='o')
plt.xticks(range(2,20))
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. Number of Clusters (Ward Linkage)")
plt.grid(True)
plt.savefig('./images/scaled_ss_score_plot.svg')
plt.show()


# In[25]:


# add the clusters labels to the dataset
hcluster_K2_labels = fcluster(wardlink,t=2, criterion='maxclust')
hcluster_K5_labels = fcluster(wardlink,t=5, criterion='maxclust')
ads_data['hcluster_k2'] = hcluster_K2_labels
ads_data['hcluster_k5'] = hcluster_K5_labels
ads_data.head()
ads_data['hcluster_k2'].value_counts()
ads_data['hcluster_k5'].value_counts()


# #### K-means Clustering
# ##### Apply K-means Clustering
# Applied the Kmeans clusting to the dataset for value os k=2 and random_state=123, got an Inertia value of 142207.
# But this does not tell if this is the ideal number of clusters using k-means, we will get the Inertia value for different values of k and plot to get the Elbow Curve.
# ##### Plot the Elbow curve 
# Plotting the elbow curve we see that the break of the elbow is seen when K=6, we will also below Check the Silhouette Scores
# ##### Check Silhouette Scores 
# Doing this we see that it suggests that 2, 10, 11, 12, 13 clusters would be optimal with 2 having more clear boundaries between the clusters than the rest, since 10, 11, 12, 13 Clusters would be too many clusters for this exercise we will explore 2 clusters, but investigate the larger clusters at a later point if needed
# ##### Figure out the appropriate number of clusters 
# We see that Silhouette Scores suggests that 2, 10, 11, 12, 13 clusters would be optimal with 2 having more clear boundaries between the clusters than the rest.
# and the Elbow method suggests we look at k = 6.
# So we will investigate k=2 and k=6 as the apporpriate number of clusters to investigate.
# ##### Cluster Profiling
# When we see the clusters created by the two methods hierarchical and k-means we see that the Clusters for k=2 has an overlap, for this exercise we will concentrate on this, it would be good to explore the other two at a later point for the business.
# Looking at the Clusters we have the following:
# Cluster 1 has lower "Ad Size , Fee, CTR, CPM", Higer "Impressions , Clicks , Spend, Revenue, CPC": Small_Ad_less_CPM_more_Revenue
# Cluster 2 has Higher "Ad Size , Fee, CTR, CPM", Lower "Impressions , Clicks , Spend, Revenue, CPC": Large_Ad_High_CPM_less_Revenue

# In[26]:


# Apply the K-means Clustering lets apply using k=2 for now
kmeans = KMeans(n_clusters = 2, random_state=123)
kmeans.fit(ads_cluster_data)


# In[27]:


#Get the inertia values for k = 2
kmeans.inertia_


# In[28]:


random_state=123
ssw = []
for i in range(2,21):
    KM = KMeans(n_clusters = i, random_state=random_state)
    KM.fit(ads_cluster_data)
    ssw.append(KM.inertia_)


# In[29]:


#plotting the Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(range(2,21), ssw,marker='o')
plt.xticks(range(2,21))
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within Clusters Variance(Initia)")
plt.title("Within Clusters Variance(Initia) vs. Number of Clusters (Kmeans)")
plt.grid(True)
plt.savefig('./images/scaled_elbow_plot.svg')
plt.show()


# In[30]:


# finding the differrent siloutte scorce at different value of k and plotting the same
ss_scores = []
for k in range(2,21):
    KM = KMeans(n_clusters = k, random_state=random_state)
    KM.fit(ads_cluster_data)    
    labels = KM.labels_
    sil = ss(ads_cluster_data,labels)
    ss_scores.append(sil)
    print(f"Silhouette Score (K={k}): {sil:.4f}")


# In[31]:


# Plotting Silhouette Scores to check which one would be optimal
# Plot the silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(range(2,21),ss_scores, marker='o')
plt.xticks(range(2,21))
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. Number of Clusters (Kmeans)")
plt.grid(True)
plt.savefig('./images/scaled_ss_score_kmean_plot.svg')
plt.show()


# In[32]:


# add the clusters labels to the dataset
KM = KMeans(n_clusters = 2, random_state=random_state)
KM.fit(ads_cluster_data)    
kmeans_K2_labels = KM.labels_

KM = KMeans(n_clusters = 6, random_state=random_state)
KM.fit(ads_cluster_data)    
kmeans_K6_labels = KM.labels_

ads_data['kmeans_k2'] = kmeans_K2_labels
ads_data['kmeans_k6'] = kmeans_K6_labels
ads_data.head()

ads_data['kmeans_k6'].value_counts()
ads_data['kmeans_k2'].value_counts()


# In[33]:


# Check the Cluster profiling
# Choose and create the final dataset for clustering analysis.
cols = ['Ad Size', 'Impressions', 'Clicks', 'Spend', 'Fee', 'Revenue', 'CTR', 'CPM', 'CPC','hcluster_k2','hcluster_k5','kmeans_k2','kmeans_k6']
cluster_profile_data = ads_data[cols]
cluster_profile_data.head()
hcluster_k2_cols = ['Ad Size', 'Impressions', 'Clicks', 'Spend', 'Fee', 'Revenue', 'CTR', 'CPM', 'CPC','hcluster_k2']
hcluster_k5_cols = ['Ad Size', 'Impressions', 'Clicks', 'Spend', 'Fee', 'Revenue', 'CTR', 'CPM', 'CPC','hcluster_k5']
kmeans_k2_cols = ['Ad Size', 'Impressions', 'Clicks', 'Spend', 'Fee', 'Revenue', 'CTR', 'CPM', 'CPC','kmeans_k2']
kmeans_k6_cols = ['Ad Size', 'Impressions', 'Clicks', 'Spend', 'Fee', 'Revenue', 'CTR', 'CPM', 'CPC','kmeans_k6']

# hcluster k=2 profiling
hcluster_k2_data = cluster_profile_data[hcluster_k2_cols]
print('Cluster Profiling hcluster k=2')
print(f'Cluster 1 has lower "Ad Size , Fee, CTR, CPM", Higer "Impressions , Clicks , Spend, Revenue, CPC"')
print(f'Cluster 2 has Higher "Ad Size , Fee, CTR, CPM", Lower "Impressions , Clicks , Spend, Revenue, CPC"')
hcluster_k2_data.groupby('hcluster_k2').mean().round(2)

# Kmeans k=2 profiling
kmeans_k5_data = cluster_profile_data[kmeans_k2_cols]
kmeans_k5_data.groupby('kmeans_k2').mean().round(2)
print('Cluster Profiling Kmeans k=2')
print(f'Cluster 0 has lower "Ad Size , Fee, CTR, CPM", Higer "Impressions , Clicks , Spend, Revenue, CPC"')
print(f'Cluster 1 has Higher "Ad Size , Fee, CTR, CPM", Lower "Impressions , Clicks , Spend, Revenue, CPC"')
kmeans_k5_data.groupby('kmeans_k2').mean().round(2)


# #### Actionable Insights & Recommendations
# 1. There is clearly two groups : Small_Ad_less_CPM_more_Revenue and Large_Ad_more_CPM_less_Revenue
# 2. More effective type of ads by size would be smaller ads with sizes around 72276.
# 3. Inorder to effect Revenue better, should consider the avenues where the Fee is 0.25 or lessser.
# 4. A Marketing Strategy to move form "Large_Ad_more_CPM_less_Revenue" segments to "Small_Ad_less_CPM_more_Revenue" segements would be benificial.
# 5. We should study the "Small_Ad_less_CPM_more_Revenue" more closely along with the mix of 'InventoryType' 'Ad Type' 'Platform' 'Device Type' 'Format' which make up this category to see how different it is from the "Large_Ad_more_CPM_less_Revenue" Group to help us move ads to the 1st group.

# ## Problem 2 - PCA

# ### Overview
# We have been given India Census data, the said data set has too many features to find useful details, We are tasked to use the Principal Components Analysis Statistical technique, to reduce the number of features be able to extract meaningfull information.

# ### Dataset
# The data set given has 61 Features and 640 observations, the data is by state and district code.
# All the features in the data are numeric expect, State and Area Name.
# | Name| Description |
# | --- | --- |
# | State | State Code |
# | District | District Code |
# | Name | Name |
# | TRU1 | Area Name |
# | No_HH | No of Household |
# | TOT_M | Total population Male |
# | TOT_F | Total population Female |
# | M_06 | Population in the age group 0-6 Male |
# | F_06 | Population in the age group 0-6 Female |
# | M_SC | Scheduled Castes population Male |
# | F_SC | Scheduled Castes population Female |
# | M_ST | Scheduled Tribes population Male |
# | F_ST | Scheduled Tribes population Female |
# | M_LIT | Literates population Male |
# | F_LIT | Literates population Female |
# | M_ILL | Illiterate Male |
# | F_ILL | Illiterate Female |
# | TOT_WORK_M | Total Worker Population Male |
# | TOT_WORK_F | Total Worker Population Female |
# | MAINWORK_M | Main Working Population Male |
# | MAINWORK_F | Main Working Population Female |
# | MAIN_CL_M | Main Cultivator Population Male |
# | MAIN_CL_F | Main Cultivator Population Female |
# | MAIN_AL_M | Main Agricultural Labourers Population Male |
# | MAIN_AL_F | Main Agricultural Labourers Population Female |
# | MAIN_HH_M | Main Household Industries Population Male |
# | MAIN_HH_F | Main Household Industries Population Female |
# | MAIN_OT_M | Main Other Workers Population Male |
# | MAIN_OT_F | Main Other Workers Population Female |
# | MARGWORK_M | Marginal Worker Population Male |
# | MARGWORK_F | Marginal Worker Population Female |
# | MARG_CL_M | Marginal Cultivator Population Male |
# | MARG_CL_F | Marginal Cultivator Population Female |
# | MARG_AL_M | Marginal Agriculture Labourers Population Male |
# | MARG_AL_F | Marginal Agriculture Labourers Population Female |
# | MARG_HH_M | Marginal Household Industries Population Male |
# | MARG_HH_F	| Marginal Household Industries Population Female |
# | MARG_OT_M | Marginal Other Workers Population Male |
# | MARG_OT_F | Marginal Other Workers Population Female |
# | MARGWORK_3_6_M | Marginal Worker Population 3-6 Male |
# | MARGWORK_3_6_F | Marginal Worker Population 3-6 Female |
# | MARG_CL_3_6_M | Marginal Cultivator Population 3-6 Male |
# | MARG_CL_3_6_F | Marginal Cultivator Population 3-6 Female |
# | MARG_AL_3_6_M | Marginal Agriculture Labourers Population 3-6 Male |
# | MARG_AL_3_6_F | Marginal Agriculture Labourers Population 3-6 Female |
# | MARG_HH_3_6_M | Marginal Household Industries Population 3-6 Male |
# | MARG_HH_3_6_F | Marginal Household Industries Population 3-6 Female |
# | MARG_OT_3_6_M | Marginal Other Workers Population Person 3-6 Male |
# | MARG_OT_3_6_F | Marginal Other Workers Population Person 3-6 Female |
# | MARGWORK_0_3_M | Marginal Worker Population 0-3 Male |
# | MARGWORK_0_3_F | Marginal Worker Population 0-3 Female |
# | MARG_CL_0_3_M | Marginal Cultivator Population 0-3 Male |
# | MARG_CL_0_3_F | Marginal Cultivator Population 0-3 Female |
# | MARG_AL_0_3_M | Marginal Agriculture Labourers Population 0 -3 Male |
# | MARG_AL_0_3_F | Marginal Agriculture Labourers Population 0-3 Female
# | MARG_HH_0_3_M | Marginal Household Industries Population 0-3 Male
# | MARG_HH_0_3_F | Marginal Household Industries Population 0-3 Female
# | MARG_OT_0_3_M | Marginal Other Workers Population 0-3 Male
# | MARG_OT_0_3_F | Marginal Other Workers Population 0-3 Female
# | NON_WORK_M | Non Working Population Male
# | NON_WORK_F | Non Working Population Female

# ### Objective
# We are tasked to use the Principal Components Analysis Statistical technique, to reduce the number of features be able to extract meaningfull information.

# ### Questions Asked

# ####  Define the problem and perform Exploratory Data Analysis
# #####  Problem Definition
# We have been given India Census data, the said data set has too many features to find useful details, We are tasked to use the Principal Components Analysis Statistical technique, to reduce the number of features be able to extract meaningfull information.
# #####  Check shape, Data types
# The dataset provided has 640 observations with 61 features.
# The dataset has 2 columns with string values, 59 columns with int values.
# ##### statistical summary - Perform an EDA on the data to extract useful insights
# ***"Note: 1. Pick 5 variables out of the given 24 variables below for EDA: No_HH, TOT_M, TOT_F, M_06, F_06, M_SC, F_SC, M_ST, F_ST, M_LIT, F_LIT, M_ILL, F_ILL, TOT_WORK_M, TOT_WORK_F, MAINWORK_M, MAINWORK_F, MAIN_CL_M, MAIN_CL_F, MAIN_AL_M, MAIN_AL_F, MAIN_HH_M, MAIN_HH_F, MAIN_OT_M, MAIN_OT_F"***
# 
# I will pick No_HH, TOT_M, TOT_F, M_06, F_06
# Checking the above 5 variables this is what we find.
# 1. State with the most households is Uttar Pradesh with ~4 million households which ~12% of the total households, State with the least households is Dadara & Nagar Havelli with 4288 households making up ~0.013% of the total households.
# 2. District with the most households is North Twenty Four Parganas in West Bengal with ~300K households which is abut .94% of the overall households, District with the least number of households is the Dibang Valley in Arunachal Pradesh with 350 households making up 0.001% of the total households
# 3. Since the Dataset is an "abstract for female headed households", the Gender ratio= TOT_F/TOT_M is higher than 100% for all states.
# 4. State with the highest Gender Ratio is Andra Pradesh with the GR at ~1.9, State with the least Gender Ratio is lakshadweep with the GR at ~1.2
# 5. District "['Krishna']" from the State "['Andhra Pradesh']"has the most Gender Ratio with the GR ~2.3, District "['Lakshadweep']" from the State "['Lakshadweep']"has the least Gender Ratiowith the GR at ~1.2.
# 6. State with the highest Child Gender Ratio (CGR) = F_06/M_06 is Arunachal Pradesh with CGR ~1.1, State with the least CGR is Haryana with CGR ~0.9
# 7. District "['East Kameng']" from the State "['Arunachal Pradesh']"has the most Child Gender Ratio with CGR ~1.2,District "['Samba']" from the State "['Jammu & Kashmir']"has the least Gender Ratio with CGR ~0.8.
# 

# In[35]:


census_data = pd.read_excel('./data/PCA_India_Data_Census.xlsx')
census_data.head()
#create a copy of the data 
census_data_copy = census_data.copy()


# In[36]:


census_data.shape


# In[37]:


census_data.info()


# In[38]:


# HH by State
eda_cols = ['State','Area Name','No_HH','TOT_M','TOT_F','M_06','F_06']
eda_data = census_data[eda_cols]
HH_sum = eda_data.groupby('State')['No_HH'].sum().sort_values(ascending=False).reset_index().rename(columns={'No_HH': 'Tot_HH'})
HH_sum['percent'] = round((HH_sum.Tot_HH/HH_sum.Tot_HH.sum())*100,3)
HH_sum.set_index(keys='State',inplace=True)

fig, axes= plt.subplots(nrows=1,ncols=2,figsize=(20,15))
sns.barplot(data=HH_sum,x='Tot_HH',y=HH_sum.index.values,ax=axes[0])
for i in axes[0].containers:
    axes[0].bar_label(i, fmt='%.0f')

axes[0].set_xticklabels(['{:,.0f}'.format(x) for x in axes[0].get_xticks()])
axes[0].set_xlabel('Total Households')
axes[0].set_ylabel('State')
axes[0].set_title('Total Households by State')

sns.barplot(data=HH_sum,x='percent',y=HH_sum.index.values,ax=axes[1])
for i in axes[1].containers:
    axes[1].bar_label(i, fmt='%.3f')

axes[1].set_xlabel('Percentage of Total Households')
axes[1].set_title('Percentage of Total Households by State')
axes[1].set_yticks([])

fig.suptitle('Households by State')
plt.tight_layout()
plt.savefig('./images/cencus_tot_HH.svg')
plt.show()


# In[39]:


# HH by Area/District
eda_cols = ['State','Area Name','No_HH','TOT_M','TOT_F','M_06','F_06']
eda_data = census_data[eda_cols]
HH_sum = eda_data.groupby('Area Name')['No_HH'].sum().sort_values(ascending=False).reset_index().rename(columns={'No_HH': 'Tot_HH'})
HH_sum['percent'] = round((HH_sum.Tot_HH/HH_sum.Tot_HH.sum())*100,3)
HH_sum.set_index(keys='Area Name',inplace=True)
HH_head = HH_sum.head(10)
HH_tail = HH_sum.tail(10)

fig, axes= plt.subplots(nrows=2,ncols=2,figsize=(20,20))
sns.barplot(data=HH_head,x='Tot_HH',y=HH_head.index.values,ax=axes[0,0])
for i in axes[0,0].containers:
    axes[0,0].bar_label(i, fmt='%.0f')

axes[0,0].set_xticklabels(['{:,.0f}'.format(x) for x in axes[0,0].get_xticks()])
axes[0,0].set_xlabel('Total Households')
axes[0,0].set_ylabel('District/Area')
axes[0,0].set_title('Top 10 Total Households by District/Area')

sns.barplot(data=HH_head,x='percent',y=HH_head.index.values,ax=axes[0,1])
for i in axes[0,1].containers:
    axes[0,1].bar_label(i, fmt='%.3f')

axes[0,1].set_xlabel('Percentage of Total Households')
axes[0,1].set_title('Top 10 Percentage of Total Households by District/Area')
axes[0,1].set_yticks([])

sns.barplot(data=HH_tail,x='Tot_HH',y=HH_tail.index.values,ax=axes[1,0])
for i in axes[1,0].containers:
    axes[1,0].bar_label(i, fmt='%.0f')

axes[1,0].set_xticklabels(['{:,.0f}'.format(x) for x in axes[0,0].get_xticks()])
axes[1,0].set_xlabel('Total Households')
axes[1,0].set_ylabel('District/Area')
axes[1,0].set_title('Least 10 Total Households by District/Area')

sns.barplot(data=HH_tail,x='percent',y=HH_tail.index.values,ax=axes[1,1])
for i in axes[1,1].containers:
    axes[1,1].bar_label(i, fmt='%.3f')

axes[1,1].set_xlabel('Percentage of Total Households')
axes[1,1].set_title('Least 10 Percentage of Total Households by District/Area')
axes[1,1].set_yticks([])

fig.suptitle('Households by District/Area')
plt.tight_layout()
plt.savefig('./images/cencus_area_tot_HH.svg')
plt.show()
Top_District = eda_data[eda_data['Area Name']=='North Twenty Four Parganas'][['State','Area Name']]
Least_District =  eda_data[eda_data['Area Name']=='Dibang Valley'][['State','Area Name']]
print(f'District "{Top_District["Area Name"].values}" from the State "{Top_District["State"].values}" has the most Child Gender Ratio.')
print(f'District "{Least_District["Area Name"].values}" from the State "{Least_District["State"].values}" has the least Child Gender Ratio.')


# In[40]:


state_male_female_data = eda_data.groupby('State')[['TOT_M','TOT_F']].sum()
state_male_female_data['Gender_Ratio'] = state_male_female_data['TOT_F']/state_male_female_data['TOT_M']
state_male_female_data_sorted = state_male_female_data.sort_values(by='Gender_Ratio', ascending=False)
state_male_female_data_sorted_head =  state_male_female_data_sorted.head(10)
state_male_female_data_sorted_tail =  state_male_female_data_sorted.tail(10)

fig, axes= plt.subplots(nrows=1,ncols=2,figsize=(20,20))
sns.barplot(data=state_male_female_data_sorted_head,x='Gender_Ratio',y=state_male_female_data_sorted_head.index.values,ax=axes[0])
for i in axes[0].containers:
    axes[0].bar_label(i, fmt='%.3f')

axes[0].set_xlabel('Gender Ratio')
axes[0].set_ylabel('State')
axes[0].set_title('Top 10 Gender Ratio by State')

sns.barplot(data=state_male_female_data_sorted_tail,x='Gender_Ratio',y=state_male_female_data_sorted_tail.index.values,ax=axes[1])
for i in axes[1].containers:
    axes[1].bar_label(i, fmt='%.3f')

axes[1].set_xlabel('Gender Ratio')
axes[1].set_title('Least 10 Gender Ratio by State')

fig.suptitle('Gender Ratio by State')
plt.tight_layout()
plt.savefig('./images/cencus_state_GR.svg')
plt.show()


# In[44]:


dist_male_female_data = eda_data.groupby('Area Name')[['TOT_M','TOT_F']].sum()
dist_male_female_data['Gender_Ratio'] = dist_male_female_data['TOT_F']/dist_male_female_data['TOT_M']
dist_male_female_data_sorted = dist_male_female_data.sort_values(by='Gender_Ratio', ascending=False)
dist_male_female_data_sorted_head =  dist_male_female_data_sorted.head(10)
dist_male_female_data_sorted_tail =  dist_male_female_data_sorted.tail(10)

fig, axes= plt.subplots(nrows=1,ncols=2,figsize=(20,20))
sns.barplot(data=dist_male_female_data_sorted_head,x='Gender_Ratio',y=dist_male_female_data_sorted_head.index.values,ax=axes[0])
for i in axes[0].containers:
    axes[0].bar_label(i, fmt='%.3f')

axes[0].set_xlabel('Gender Ratio')
axes[0].set_ylabel('State')
axes[0].set_title('Top 10 Gender Ratio by District')

sns.barplot(data=dist_male_female_data_sorted_tail,x='Gender_Ratio',y=dist_male_female_data_sorted_tail.index.values,ax=axes[1])
for i in axes[1].containers:
    axes[1].bar_label(i, fmt='%.3f')

axes[1].set_xlabel('Gender Ratio')
axes[1].set_title('Least 10 Gender Ratio by District')

fig.suptitle('Gender Ratio by District')
plt.tight_layout()
plt.savefig('./images/cencus_dist_GR.svg')
plt.show()

Top_District = eda_data[eda_data['Area Name']=='Krishna'][['State','Area Name']]
Least_District =  eda_data[eda_data['Area Name']=='Lakshadweep'][['State','Area Name']]
print(f'District "{Top_District["Area Name"].values}" from the State "{Top_District["State"].values}" has the most Child Gender Ratio.')
print(f'District "{Least_District["Area Name"].values}" from the State "{Least_District["State"].values}" has the least Child Gender Ratio.')


# In[45]:


state_child_male_female_data = eda_data.groupby('State')[['M_06','F_06']].sum()
state_child_male_female_data['Child_Gender_Ratio'] = state_child_male_female_data['F_06']/state_child_male_female_data['M_06']
state_child_male_female_data_sorted = state_child_male_female_data.sort_values(by='Child_Gender_Ratio', ascending=False)
state_child_male_female_data_sorted_head =  state_child_male_female_data_sorted.head(10)
state_child_male_female_data_sorted_tail =  state_child_male_female_data_sorted.tail(10)

fig, axes= plt.subplots(nrows=1,ncols=2,figsize=(20,20))
sns.barplot(data=state_child_male_female_data_sorted_head,x='Child_Gender_Ratio',y=state_child_male_female_data_sorted_head.index.values,ax=axes[0])
for i in axes[0].containers:
    axes[0].bar_label(i, fmt='%.3f')

axes[0].set_xlabel('Child Gender Ratio')
axes[0].set_ylabel('State')
axes[0].set_title('Top 10 Child Gender Ratio by State')

sns.barplot(data=state_child_male_female_data_sorted_tail,x='Child_Gender_Ratio',y=state_child_male_female_data_sorted_tail.index.values,ax=axes[1])
for i in axes[1].containers:
    axes[1].bar_label(i, fmt='%.3f')

axes[1].set_xlabel('Child Gender Ratio')
axes[1].set_title('Least 10 Child Gender Ratio by State')

fig.suptitle('Child Gender Ratio by State')
plt.tight_layout()
plt.savefig('./images/cencus_state_CGR.svg')
plt.show()


# In[47]:


dist_child_male_female_data = eda_data.groupby('Area Name')[['M_06','F_06']].sum()
dist_child_male_female_data['Child_Gender_Ratio'] = dist_child_male_female_data['F_06']/dist_child_male_female_data['M_06']
dist_child_male_female_data_sorted = dist_child_male_female_data.sort_values(by='Child_Gender_Ratio', ascending=False)
dist_child_male_female_data_sorted_head =  dist_child_male_female_data_sorted.head(10)
dist_child_male_female_data_sorted_tail =  dist_child_male_female_data_sorted.tail(10)

fig, axes= plt.subplots(nrows=1,ncols=2,figsize=(20,20))
sns.barplot(data=dist_child_male_female_data_sorted_head,x='Child_Gender_Ratio',y=dist_child_male_female_data_sorted_head.index.values,ax=axes[0])
for i in axes[0].containers:
    axes[0].bar_label(i, fmt='%.3f')

axes[0].set_xlabel('Child Gender Ratio')
axes[0].set_ylabel('District')
axes[0].set_title('Top 10 Child Gender Ratio by District')

sns.barplot(data=dist_child_male_female_data_sorted_tail,x='Child_Gender_Ratio',y=dist_child_male_female_data_sorted_tail.index.values,ax=axes[1])
for i in axes[1].containers:
    axes[1].bar_label(i, fmt='%.3f')

axes[1].set_xlabel('Child Gender Ratio')
axes[1].set_title('Least 10 Child Gender Ratio by District')

fig.suptitle('Child Gender Ratio by District')
plt.tight_layout()
plt.savefig('./images/cencus_dist_CGR.svg')
plt.show()
Top_District = eda_data[eda_data['Area Name']=='East Kameng'][['State','Area Name']]
Least_District =  eda_data[eda_data['Area Name']=='Samba'][['State','Area Name']]
print(f'District "{Top_District["Area Name"].values}" from the State "{Top_District["State"].values}" has the most Child Gender Ratio.')
print(f'District "{Least_District["Area Name"].values}" from the State "{Least_District["State"].values}" has the least Child Gender Ratio.')


# #### Data Preprocessing
# 1. Drop the two columns with string values, "State" and "Area Name"
# ##### Check for and treat (if needed) missing values
# 1. There is no Missing values in the dataset.
# ##### Check for and treat (if needed) data irregularities
# 1. There is no Duplicates
# 2. Reviewing the data using describe, There are some features wihth min 0 values,but these could be valid entries as could be zero for said features, hence we will not treat these.
# ##### Visualize the data before and after scaling and comment on the impact on outliers
# 1. There are outliers for all the features expect State Code and District Code which is expected as State Code and District Code are identifiers rather than counts.
# ##### Scale the Data using the z-score method
# 1. We Scale the data using z-score method and plots the box plots again, I have used the Scipy zscore method we can also use the StandarScaler method from Sklearn which does the same scaling operation.
# 2. There are outliers for all the features expect State Code and District Code which is expected as State Code and District Code are identifiers rather than counts.
# 3. Scaling does not have an impact on outliers, we still see extreme outliers in the dataset.
# 4. The data set has outliers and extreme outliers,we cannot remove these as these are not due to data errors, we ideally would explore the following below:***(but would not do the same for this exercise)***
#     1. Run transformations like square root, cube root, log, or box-cox etc, to reduce the skew in the data a potientially reduce the ouliers and outlier impact
#     2. Post transformation and scaling if outliers/extreme outliers still exists, we would explore techniques like Robust PCA to make sure the impact of the outliers are minimized in the creation of the Principal Components.

# In[48]:


# Data Preprocessing to appling PCA
# Remove Odbject type cols
census_data_num = census_data.select_dtypes(exclude=['object'])
census_data_num.info()
census_data_num.isna().sum()


# In[49]:


census_data_num.duplicated().sum()


# In[50]:


census_data_num.describe().round(2).T


# In[51]:


# boxplots before scaling
census_data_num_melt=pd.melt(census_data_num,var_name='Features',value_name='Values')
plt.figure(figsize=(20,10))
sns.boxplot(census_data_num_melt,x='Features',y='Values')
plt.title('Census Data Boxplots-Before Scaling')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('./images/pca_box.svg')
plt.show()


# In[52]:


#Scaling all the numercial columns to the same scale using zscore scaling
scaled_census_data_num =zscore(census_data_num)
scaled_census_data_num.describe().round(2).T


# In[53]:


# boxplots after scaling
scaled_census_data_num_melt=pd.melt(scaled_census_data_num,var_name='Features',value_name='Values')
plt.figure(figsize=(20,10))
sns.boxplot(scaled_census_data_num_melt,x='Features',y='Values')
plt.title('Census Data Boxplots-After Scaling')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('./images/pca_box_scaled.svg')
plt.show()


# In[54]:


scaled_census_data_num.head()


# #### PCA
# Note: For the scope of this project, take at least 90% explained variance.
# ##### Create the covariance matrix
# Creating the covariance matrix using the cov() method, and plotting the same in a heatmap, we see that there are lot of instances where the value of the covariance is larger than 0.8.
# This indicates that there is alot of covariance between the features and the dataset is a good candidate for PCA.
# ##### Get eigen values and eigen vectors
# Using the PCA method of the SKlearn package we can get the eigen values (explained_variance_), the eigen vectors (components_) and the eigen values ratios (explained_variance_ratio_) 
# ##### Identify the optimum number of PCs
# We take the theshold of 90% explained variance or .90 as the threshold to get the number of PCs whic are Optimal, from the cumalative explained variance ratio we set this treshold and count the numbers of PCs below or equal to this threshold, we get a value of 6, which means that the 1st six PCs explain the variance in the data to a theashold of 90%.
# #####  Show Scree plot 
# Ploting the Scree Plot and specifically the Cumalative Scree Plot we see if we cut of the graph at 0.90 on the y axis we have 6 PCs which would get us close to this.
# ##### Compare PCs with Actual Columns and identify which is explaining most variance
# 1. Plotted the Heat map of the loadings for the Actual Columns(Feature) vs frist 6 PCs, this gives us visibility to the influence of the feature within a PC, higher the magnitude more the influence, the sign lets us know if its a positive influence or negative.
# 2. Plotted for each of the first 6 PC's the square of the loadings (tells us the contribution to variance) for the actual columns(features), this shows us for each PC, what is the features and what is they contribution in explaining the variance with in the PC, we only looked at features which explained ~0.90 or ~90% of the variance within the PC.
# 3. The list of the features explaining most of the variance in each of the 6 PCs are:
#     1. The feature explaining most that the variance for PC1 is, TOT_M, it expains 2.79% of the variance in PC1.
#     2. The feature explaining most that the variance for PC2 is, MARG_CL_M, it expains 7.29% of the variance in PC2.
#     3. The feature explaining most that the variance for PC3 is, MAIN_AL_F, it expains 9.83% of the variance in PC3.
#     4. The feature explaining most that the variance for PC4 is, MARG_AL_3_6_F, it expains 8.36% of the variance in PC4.
#     5. The feature explaining most that the variance for PC5 is, F_ST, it expains 14.3% of the variance in PC5.
#     6. The feature explaining most that the variance for PC6 is, MAIN_HH_F, it expains 18.95% of the variance in PC6.
# 

# In[55]:


covMatrix = scaled_census_data_num.cov()
matrix = np.triu(covMatrix)
plt.figure(figsize=(20, 20))
sns.heatmap(covMatrix, annot=True, fmt='.1f',mask=matrix)
plt.title('Covariance Matrix')
plt.tight_layout()
plt.savefig('./images/pca_cov.svg')
plt.show()


# In[56]:


#PCA get the eigen values and the eigenvectors
pca = PCA(n_components=scaled_census_data_num.shape[1])
pca.fit(scaled_census_data_num)
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
egenvaluesratio = pca.explained_variance_ratio_
egenvaluesratio_cumulative = pca.explained_variance_ratio_.cumsum()

#Get the Optiminum Number of PCS
threshold = 0.90
mask = egenvaluesratio_cumulative <= threshold
optimum_pcs = egenvaluesratio_cumulative[mask]
print(f'The number of optimum PCs is {len(optimum_pcs)}')


# In[57]:


#Plotting the PCA Feature Explained, Scree and Cumalitive Scree Plots

fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(20,4))


ax[0].bar(range(1, 60), pca.explained_variance_ratio_)
ax[0].set_ylabel('Explained Variance')
ax[0].set_xlabel('Principal Components')
ax[0].set_title('Explained Variance Ratio')


sns.lineplot(y=pca.explained_variance_ratio_ ,x=range(1,60),marker='o',linewidth=1,ax=ax[1])
ax[1].set_xlabel('Number of Components')
ax[1].set_ylabel('Variance Explained Ratio')
ax[1].set_title('Scree Plot')

#get the cummerlative variance ratio:
cum_ratio = np.cumsum(pca.explained_variance_ratio_)
cum_ratio

sns.lineplot(y=cum_ratio ,x=range(1,60),marker='o',ax=ax[2])
ax[2].set_xlabel('Number of Components')
ax[2].set_ylabel('Cumulative Variance Explained Ratio')
ax[2].set_title('Cumulative Scree Plot')
ax[2].axhline(y=0.9, color='r', linestyle='--')

plt.tight_layout()
plt.savefig('./images/pca_scree_plots.svg')
plt.show()
print(f'The number of optimum PCs whicn explains ~90% of the variance is {len(optimum_pcs)}.')


# In[58]:


# We will concentrate on the frist 6 Principal Components as they explain ~90% of the variance 
# Get the eigenvectors and label them to the corresponding columns of the data.
pc_labels =['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10',
            'PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20',
            'PC21','PC22','PC23','PC24','PC25','PC26','PC27','PC28','PC29','PC30',
            'PC31','PC32','PC33','PC34','PC35','PC36','PC37','PC38','PC39','PC40',
            'PC41','PC42','PC43','PC44','PC45','PC46','PC47','PC48','PC49','PC50',
            'PC51','PC52','PC53','PC54','PC55','PC56','PC57','PC58','PC59'
           ] 
loadings_actual_features = pd.DataFrame(eigenvectors.T,columns=pc_labels,index=scaled_census_data_num.columns)
first_6_component_loadings= loadings_actual_features.iloc[:,:6]
first_3_component_loadings = first_6_component_loadings.iloc[:,:3]
first_6_component_loadings


# In[59]:


#heatmap for the 1st 6 PC loadings/eigen vectors by Actual columns(Features) vs PCs
plt.figure(figsize=(10,15))
colormap = sns.color_palette("Blues")
sns.heatmap(first_6_component_loadings,annot=True, fmt='.3F', cmap=colormap)
plt.suptitle('Heat Map of Loadings by Acutal Columns by Pricinpal Components, First 6 components are choosen.',ha='center',va='center')
plt.ylabel('Acutal Columns')
plt.xlabel('Principal Components')
plt.tight_layout()
plt.savefig('./images/pca_feature_heatmap.svg')
plt.show()


# In[60]:


# ploting the loading squared(variance explained %) of the 1st 6 Principal components by actual freatures/columns. 
fig, ax = plt.subplots(nrows=6, ncols=1,figsize=(10,35))
for i, col in enumerate(first_6_component_loadings.columns):
    component_loadings_sort = first_6_component_loadings[col].pow(2).sort_values(ascending=False)
    component_loadings_sort_cumsum = component_loadings_sort.cumsum()
    mask = component_loadings_sort_cumsum <= threshold
    component_loadings_sort = round(component_loadings_sort[mask]*100,2)
    axes = ax[i]
    sns.barplot(x=component_loadings_sort.index.values,y=component_loadings_sort, ax = axes)
    axes.set_title(f'Contribution to Variance for {col}')
    axes.set_ylabel(f'Loading Squared')
    axes.set_xticklabels(axes.get_xticklabels(),rotation=90,fontsize=8)
    for p in axes.patches:
        axes.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),ha = 'center', va = 'center',xytext = (0, 10), 
                         textcoords = 'offset points',fontsize=6)
plt.suptitle('0.90 Contribution to Variance (Loadings squared) by Actual Columns (Features) for top 6 Principal Components',y=1)
plt.tight_layout()
plt.savefig('./images/pca_1st_three_loadings_squared.svg')
plt.show()


# ##### Write inferences about all the PCs in terms of actual variables
# Below is listed the Inferences of the PCs in terms of actual variables.

# In[61]:


# Indentify which feature is explainning the most variance in each or the 6 PCs
for i, col in enumerate(first_6_component_loadings.columns):
    print(f"Inferences for {col}:")
    print("="*200)
    component_loadings_sort = first_6_component_loadings[col].pow(2).sort_values(ascending=False)
    print(f"The feature explaining most that the variance for {col} is, {component_loadings_sort.index[0]}, it expains {round(component_loadings_sort.iloc[0]*100,2)}% of the variance in {col}.")
    print(f'There is a {round(first_6_component_loadings[col].loc[component_loadings_sort.index[0]]*100,3)}% Increase(if % is positive) or Decrease (if % is negative) in {col} for every one unit increase in Scaled {component_loadings_sort.index[0]}')
    print("="*200)
    print(f"Features explaining ~90% of the variance of {col} in desending order are:")
    component_loadings_sort_cumsum = component_loadings_sort.cumsum()
    mask = component_loadings_sort_cumsum <= threshold
    component_loadings_sort = round(component_loadings_sort[mask]*100,2)
    for idx,f in enumerate(component_loadings_sort):
        print(f'{component_loadings_sort.index[idx]}:{f}%: There is a {round(first_6_component_loadings[col].loc[component_loadings_sort.index[idx]]*100,3)}% Increase(if % is positive) or Decrease (if % is negative) in {col} for every one unit increase in Scaled {component_loadings_sort.index[idx]}.')


# ##### Write linear equation for first PC
# Below is given the linear equation in the format PC1 = c1*x1+c2*x2....+ci*xi, where the c1,c2,...,ci are the coffients given by the eigenvectors and x1,x2,...,xi are the values of the actual columns, e.g. 'State Code','No_HH','TOT_F','M_SC' etc.

# In[62]:


first_6_component_loadings.iloc[:,0].name
equation = f'{first_6_component_loadings.iloc[:,0].name} = '
equ_terms = []
for idx, vec in enumerate(first_6_component_loadings.iloc[:,0]):
    equ_terms.append(f'{round(vec,5)}*{first_6_component_loadings.index[idx]}')
for idx , term in enumerate(equ_terms):
    if idx == 0:
        equation = equation+" ("+term+")"
    else:
        equation = equation+" + ("+term+")"
equation


# In[ ]:




