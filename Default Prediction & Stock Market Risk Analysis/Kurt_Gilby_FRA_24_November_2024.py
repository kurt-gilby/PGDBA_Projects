#!/usr/bin/env python
# coding: utf-8

# # Part A

# ## Question 1

# ### Define the Problem
# Problem Statement
# 
# The venture capitalists aim to build a Financial Health Assessment Tool to evaluate the financial stability and creditworthiness of companies. The tool will assist in:
# 
# Debt Management Analysis: Identifying trends and assessing companies' ability to manage and fulfill their financial obligations effectively.
# Credit Risk Evaluation: Estimating the risk of default through financial metrics like liquidity ratios, debt-to-equity ratios, and other indicators.
# As a Data Scientist, your role is to develop a predictive model using the given dataset, which includes detailed financial metrics of companies. The model's objective is to classify companies as defaulters or non-defaulters, based on whether their Net Worth Next Year is positive or negative.
# 
# Key Deliverables:
# Target Variable: "Net Worth Next Year" – Positive values indicate a non-defaulter, while negative values indicate a defaulter.
# Model Inputs: Financial metrics like total assets, total liabilities, debt-to-equity ratio, and other balance sheet indicators.
# Outcome: A robust machine learning model that predicts the likelihood of default.
# 
# enabling stakeholders to:
# Identify companies at financial risk.
# Make informed decisions about debt management and investments.
# Business Impact: Facilitate proactive risk mitigation strategies and enhance the decision-making process for investors and businesses.

# In[1]:


get_ipython().run_line_magic('pip', 'install imblearn')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


# In[3]:


funda_data = pd.read_csv('./data/Company29.csv')
funda_data.head()


# ### Check shape

# In[4]:


print(f'Data has {funda_data.shape[0]} rows/observations.')
print(f'Data has {funda_data.shape[1]} columns/features.')


# ### Check Data Types

# In[5]:


funda_data.info()
print(f'50 columns/features are of data type float.')
print('One column "Num" is of datatype int and is an identfier')
print('converting column "Num" to string and making it the index')
funda_data.Num = funda_data.Num.astype(str)
funda_data = funda_data.set_index(keys='Num')
funda_data.info()
funda_data.head()


# ### Statistical Summary

# In[6]:


print('There are substantial differences in scale among variables.')
print('Equity Face value has same value in the three percentiles, the value is 10, most of the values in this column as similar')
print('There are missing values for quite a few values and we will need treat them.')
pd.set_option('display.float_format', lambda x: '%.f' % x)
funda_data.describe().T


# #### Univariate Analysis

# In[7]:


print('All the columns have outliers, we would need to treat them.')
fig, ax = plt.subplots(len(funda_data.columns), figsize=(8,100))

for i, col_val in enumerate(funda_data.columns):

    sns.boxplot(y=funda_data[col_val], ax=ax[i])
    ax[i].set_title('Box plot - {}'.format(col_val), fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)

plt.savefig('./images/Blox_plots_before_Scaling.svg')
plt.show()


# #### Multivariate Analysis

# In[8]:


funda_data.columns
print('spilt the data into managable columnsets to do pairplots')

colset1 = ['Networth Next Year', 'Total assets','Total liabilities', 'Net worth', 'Total income',
       'Change in stock', 'Total expenses', 'Profit after tax', 'PBDITA',
       'PBT', 'Cash profit']
colset2 = ['Networth Next Year','PBDITA as % of total income',
       'PBT as % of total income', 'PAT as % of total income',
       'Cash profit as % of total income', 'PAT as % of net worth']
colset3 = ['Networth Next Year', 'Sales',
       'Income from fincial services', 'Other income', 'Total capital',
       'Reserves and funds', 'Borrowings', 'Current liabilities & provisions',
       'Deferred tax liability', 'Shareholders funds',
       'Cumulative retained profits', 'Capital employed',]
colset4 = ['Networth Next Year', 'TOL/TNW',
       'Total term liabilities / tangible net worth',
       'Contingent liabilities / Net worth (%)']
colset5 = ['Networth Next Year','Contingent liabilities',
       'Net fixed assets', 'Investments', 'Current assets',
       'Net working capital']
colset6 = ['Networth Next Year','Quick ratio (times)', 'Current ratio (times)',
       'Debt to equity ratio (times)', 'Cash to current liabilities (times)',
       'Cash to average cost of sales per day','PE on BSE']
colset7 = ['Networth Next Year','Creditors turnover',
       'Debtors turnover', 'Finished goods turnover', 'WIP turnover',
       'Raw material turnover']
colset8 =['Networth Next Year','Shares outstanding', 'Equity face value',
       'EPS', 'Adjusted EPS']


# In[9]:


print('The pairpolt show weak and relations of "Networth Next Year" with the non ratio features.')
print('The pairpolt no relations of "Networth Next Year" with the ratio features.')
print('The pairpolt relations of between the non ratio and ratio features.')

sns.pairplot(funda_data[colset1], diag_kind='kde')
plt.suptitle('Pairplot', y=1.02)
plt.savefig('./images/colset1.jpg')
plt.show()
sns.pairplot(funda_data[colset2], diag_kind='kde')
plt.suptitle('Pairplot', y=1.02)
plt.savefig('./images/colset2.jpg')
plt.show()
sns.pairplot(funda_data[colset3], diag_kind='kde')
plt.suptitle('Pairplot', y=1.02)
plt.savefig('./images/colset3.jpg')
plt.show()
sns.pairplot(funda_data[colset4], diag_kind='kde')
plt.suptitle('Pairplot', y=1.02)
plt.savefig('./images/colset4.jpg')
plt.show()
sns.pairplot(funda_data[colset5], diag_kind='kde')
plt.suptitle('Pairplot', y=1.02)
plt.savefig('./images/colset5.jpg')
plt.show()
sns.pairplot(funda_data[colset6], diag_kind='kde')
plt.suptitle('Pairplot', y=1.02)
plt.savefig('./images/colset6.jpg')
plt.show()
sns.pairplot(funda_data[colset7], diag_kind='kde')
plt.suptitle('Pairplot', y=1.02)
plt.savefig('./images/colset7.jpg')
plt.show()
sns.pairplot(funda_data[colset8], diag_kind='kde')
plt.suptitle('Pairplot', y=1.02)
plt.savefig('./images/colset8.jpg')
plt.show()


# In[10]:


print('The heatmap shows high corelation of "Networth Next Year", with ~19 features.')
print('The heatmap shows high corelation between a features which points us in the direction of the need for feature elimination using techinque like pvalue or VIF')
# Correlation heatmap
plt.figure(figsize=(15, 10))
corr_matrix = funda_data.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig('./images/Heatmap.svg')
plt.show()


# ### Key meaningful observations
# - There are substantial differences in scale among variables.
# - Equity Face value has same value in the three percentiles, the value is 10, most of the values in this column as similar.
# - There are missing values for quite a few values and we will need treat them.
# - All the columns have outliers, we would need to treat them.
# - The pairpolt show weak and relations of "Networth Next Year" with the non ratio features.
# - The pairpolt no relations of "Networth Next Year" with the ratio features.
# - The pairpolt relations of between the non ratio and ratio features.
# - The heatmap shows high corelation of "Networth Next Year", with ~19 features.
# - The heatmap shows high corelation between a features which points us in the direction of the need for feature elimination using techinque like pvalue or VIF.
# 

# ## Question 2

# ### Data Pre-processing

# #### Missing Value treatment

# In[11]:


funda_data.isnull().sum().sum()
print(f'There are {funda_data.isnull().sum().sum()} missing values of total {funda_data.shape[0]*funda_data.shape[1]} values.')
print(f'Percentage of missing = {(17778/212800)*100}')
print(f'There are {funda_data[funda_data.isnull().sum(axis=1) > 5].isnull().sum(axis=1).count()} rows with great than 5 missing values.')
print(f'Percentage of rows missing more than 5 values = {(1062 /4256)*100}')
print('A lot of values are missing we will use KNN to impute missing values')


# ##### KNN Imputer

# In[12]:


knn_imputer = KNNImputer(n_neighbors=5)
funda_data_imputted = pd.DataFrame(knn_imputer.fit_transform(funda_data), columns=funda_data.columns)
funda_data_imputted.isnull().sum()


# #### Outlier treatment
# All Columns/Features have outliers
# We will treat as we want to go Logistic regression in addition to other classification models.

# In[13]:


# Checking for outliers
def check_outlier(col):
    sorted(col)
    Q1,Q3=col.quantile([.25,.75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


# In[14]:


num=0
for col in funda_data_imputted.columns:
    num+=1
    lower_range, upper_range = check_outlier(funda_data_imputted[col])
    print(num,lower_range, upper_range)


# In[15]:


def treat_outlier(x):
    # taking 5,25,75 percentile of column
    q5= np.percentile(x,5)
    q25=np.percentile(x,25)
    q75=np.percentile(x,75)
    q95=np.percentile(x,95)
    #calculationg IQR range
    IQR=q75-q25
    #Calculating minimum threshold
    lower_bound=q25-(1.5*IQR)
    upper_bound=q75+(1.5*IQR)
    #Calculating maximum threshold
    #print("5th Percentile:", q5)
    #print("25th Percentile:", q25)
    #print("75th Percentile:", q75)
    #print("95th Percentile:", q95)
    #print("Lower Bound:", lower_bound)
    #print("Upper Bound:", upper_bound)
    ##Capping outliers
    return x.apply(lambda y: q75 if y > upper_bound else y).apply(lambda y: q25 if y < lower_bound else y)


# In[16]:


for col in funda_data_imputted.columns:
    funda_data_imputted[col] = treat_outlier(funda_data_imputted[col])


# In[17]:


fig, ax = plt.subplots(len(funda_data_imputted.columns), figsize=(8,100))

for i, col_val in enumerate(funda_data_imputted.columns):

    sns.boxplot(y=funda_data_imputted[col_val], ax=ax[i])
    ax[i].set_title('Box plot - {}'.format(col_val), fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)
plt.savefig('./images/Box_plots_after_treatment.svg')
plt.show()


# #### Spilt Data
# Create the Defaulter Traget column
# Split the data
# They Defaulter column is less then 30% we will use straitify

# In[18]:


#Create the Default Column 
funda_data_imputted['Defaulter'] = np.where(funda_data_imputted['Networth Next Year'] <= 0, 1, 0)
#check the % split of the defaulters
funda_data_imputted['Defaulter'].value_counts(normalize=True)*100
#spilt data in 70 30 ratio of train and test 
X = funda_data_imputted.drop(['Defaulter','Networth Next Year'], axis=1).copy()
y = funda_data_imputted['Defaulter'].copy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=123,stratify=funda_data_imputted['Defaulter'])


# #### Scale Data
# Scaling data as  we want to do Logistic regression in addition to other classification models.
# Combine the X and y as will need need them to be together to do supervised classification models. 

# In[19]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Question 3

# ### Model Building
# We will buil two models Logistic Regression and Radom Forest.
# We will check the performance of the two models using metrics of Recall primary as this is Defaulter Classification and compare the other metrics like accuracy, precision etc. once a good recall is found.

# In[20]:


log_reg = LogisticRegression(random_state=123, max_iter = 1000)
log_reg.fit(X_train_scaled, y_train)

y_train_pred_log_reg = log_reg.predict(X_train_scaled)
y_train_pred_log_reg_prob = log_reg.predict_proba(X_train_scaled)[:, 1]
y_test_pred_log_reg = log_reg.predict(X_test_scaled)
y_test_pred_log_reg_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train_scaled, y_train)

y_train_pred_rf = rf.predict(X_train_scaled)
y_train_pred_rf_prob = rf.predict_proba(X_train_scaled)[:, 1]
y_test_pred_rf = rf.predict(X_test_scaled)
y_test_pred_rf_prob = rf.predict_proba(X_test_scaled)[:, 1]


# In[21]:


cm_log_reg_train = confusion_matrix(y_train, y_train_pred_log_reg)
cm_log_reg_test = confusion_matrix(y_test, y_test_pred_log_reg)

print('Logistic Regression Train Results')
disp_cm_log_reg_train = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg_train)
disp_cm_log_reg_train.plot()
print(classification_report(y_train,y_train_pred_log_reg))


# In[22]:


print('Logistic Regression Test Results')
cm_log_reg_test = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg_train)
cm_log_reg_test.plot()
print(classification_report(y_test,y_test_pred_log_reg))


# In[23]:


cm_rf_train = confusion_matrix(y_train, y_train_pred_rf)
cm_rf_test = confusion_matrix(y_test, y_test_pred_rf)

print('Random Forest Train Results')
disp_cm_rf_train = ConfusionMatrixDisplay(confusion_matrix=cm_rf_train)
disp_cm_rf_train.plot()
print(classification_report(y_train,y_train_pred_rf))


# In[24]:


print('Random Forest Test Results')
disp_cm_rf_train = ConfusionMatrixDisplay(confusion_matrix=cm_rf_train)
disp_cm_rf_train.plot()
print(classification_report(y_test,y_test_pred_rf))


# #### Observations
# - The Logistic Regresstion model does not perform good with both the train and test data set with the  recall for "Defaulters" only at 0.01 for both.
# - The Random Forest model does better performance on train with recall for "Defaulters" only at 0.74 but fails in test with a recall of only 0.10, which tells us it is over fitting

# ## Question 4

# ### Model Performance Improvement
# - We will try to improve the both the models with using methods like 
#     - VIF
#     - PValue method
#     - Optimal Threshold for Logistic Regression and parameter tunning for Random Forest

# #### VIF (Variance Inflation Factor)

# In[25]:


def compute_vif(dataframe):
    """
    Compute VIF for each feature in a Pandas DataFrame.
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = dataframe.columns
    vif_data["VIF"] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
    return vif_data


# In[26]:


# Select numeric features
vif_X_train_scaled = pd.DataFrame(X_train_scaled,columns=X_train.columns, index=X_train.index).copy()
# Remove Equity Face Value
vif_X_train_scaled.drop(columns='Equity face value',inplace=True)
vif_result = compute_vif(vif_X_train_scaled)
vif_result.sort_values(by='VIF', ascending=False)


# In[27]:


# removing Total assets 
vif_X_train_scaled.drop(columns='Total assets',inplace=True)
# Select numeric features
vif_result = compute_vif(vif_X_train_scaled)
vif_result.sort_values(by='VIF', ascending=False)


# In[28]:


# removing Sales 
vif_X_train_scaled.drop(columns='Sales',inplace=True)
# Select numeric features
vif_result = compute_vif(vif_X_train_scaled)
vif_result.sort_values(by='VIF', ascending=False)


# In[29]:


# removing Total Income 
vif_X_train_scaled.drop(columns='Total income',inplace=True)
# Select numeric features
vif_result = compute_vif(vif_X_train_scaled)
vif_result.sort_values(by='VIF', ascending=False)


# In[30]:


# removing Net Worth
vif_X_train_scaled.drop(columns='Net worth',inplace=True)
# Select numeric features
vif_result = compute_vif(vif_X_train_scaled)
vif_result.sort_values(by='VIF', ascending=False)


# In[31]:


# removing Total liabilities
vif_X_train_scaled.drop(columns='Total liabilities',inplace=True)
# Select numeric features
vif_result = compute_vif(vif_X_train_scaled)
vif_result.sort_values(by='VIF', ascending=False)


# In[32]:


# removing PBT
vif_X_train_scaled.drop(columns='PBT',inplace=True)
# Select numeric features
vif_result = compute_vif(vif_X_train_scaled)
vif_result.sort_values(by='VIF', ascending=False)


# In[33]:


# removing PBDITA
vif_X_train_scaled.drop(columns='PBDITA',inplace=True)
# Select numeric features
vif_result = compute_vif(vif_X_train_scaled)
vif_result.sort_values(by='VIF', ascending=False)


# In[34]:


# removing PBT as % of total income
vif_X_train_scaled.drop(columns='PBT as % of total income',inplace=True)
# Select numeric features
vif_result = compute_vif(vif_X_train_scaled)
vif_result.sort_values(by='VIF', ascending=False)


# In[35]:


# removing Capital employed
vif_X_train_scaled.drop(columns='Capital employed',inplace=True)
# Select numeric features
vif_result = compute_vif(vif_X_train_scaled)
vif_result.sort_values(by='VIF', ascending=False)


# In[36]:


# removing Current assets
vif_X_train_scaled.drop(columns='Current assets',inplace=True)
# Select numeric features
vif_result = compute_vif(vif_X_train_scaled)
vif_result.sort_values(by='VIF', ascending=False)


# In[37]:


X_train_scaled = pd.DataFrame(X_train_scaled,columns=X_train.columns, index=X_train.index).copy()
X_train_scaled = X_train_scaled[vif_X_train_scaled.columns].copy()


# #### The VIF helps drop 11 Features for a thresold of 5, 
# - Equity Face values
# - Total Assets
# - Sales
# - Total Income
# - Net Worth
# - Total Liabilities
# - PBT
# - -PBDITA
# - PBT as a % of Total Income
# - Capital Employed
# - Current Assets

# #### Pvalue Selection

# In[38]:


X_with_const = sm.add_constant(X_train_scaled).copy()


# In[39]:


logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[40]:


# Removing p-values of features: Change in stock  
X_with_const.drop(columns='Change in stock',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[41]:


# Removing p-values of features: PAT as % of net worth  
X_with_const.drop(columns='PAT as % of net worth',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[42]:


# Removing p-values of features: PBDITA as % of total income   
X_with_const.drop(columns='PBDITA as % of total income',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[43]:


# Removing p-values of features: Quick ratio (times)  
X_with_const.drop(columns='Quick ratio (times)',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[44]:


# Removing p-values of features: Net working capital 
X_with_const.drop(columns='Net working capital',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[45]:


# Removing p-values of features: Cumulative retained profits  
X_with_const.drop(columns='Cumulative retained profits',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[46]:


# Removing p-values of features: EPS                 
X_with_const.drop(columns='EPS',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[47]:


# Removing p-values of features: Borrowings                                                     
X_with_const.drop(columns='Borrowings',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[48]:


# Removing p-values of features: Net fixed assets                                                     
X_with_const.drop(columns='Net fixed assets',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[49]:


# Removing p-values of features: Contingent liabilities / Net worth (%)                                                   
X_with_const.drop(columns='Contingent liabilities / Net worth (%)',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[50]:


# Removing p-values of features: Total capital                                                   
X_with_const.drop(columns='Total capital',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[51]:


# Removing p-values of features: Deferred tax liability                                                     
X_with_const.drop(columns='Deferred tax liability',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[52]:


# Removing p-values of features: Income from fincial services                                                    
X_with_const.drop(columns='Income from fincial services',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[53]:


# Removing p-values of features: Cash profit                                                    
X_with_const.drop(columns='Cash profit',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[54]:


# Removing p-values of features: Finished goods turnover                                                   
X_with_const.drop(columns='Finished goods turnover',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[55]:


# Removing p-values of features: WIP turnover                                                   
X_with_const.drop(columns='WIP turnover',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[56]:


# Removing p-values of features: Other income                                                    
X_with_const.drop(columns='Other income',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[57]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='Debtors turnover',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[58]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='Contingent liabilities',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[59]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='Total term liabilities / tangible net worth',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[60]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='TOL/TNW',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[61]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='Creditors turnover',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[62]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='Shares outstanding',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[63]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='PE on BSE',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[64]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='PAT as % of total income',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[65]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='Profit after tax',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[66]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='Investments',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[67]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='Cash to current liabilities (times)',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[68]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='Cash to average cost of sales per day',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[69]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='Adjusted EPS',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[70]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='Debt to equity ratio (times)',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[71]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='Current liabilities & provisions',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# In[72]:


# Removing High p-values of features                                                    
X_with_const.drop(columns='Shareholders funds',inplace=True)

logit_model = sm.Logit(y_train, X_with_const)
result = logit_model.fit(disp=False)  # Suppress output during fitting
    
# Get p-values for all features
p_values = result.pvalues
print("p-values of features:")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(p_values.sort_values(ascending = False))


# #### After Pvalue Selection we have 5 Features

# In[73]:


X_with_const.drop(columns='const',inplace=True)


# In[74]:


X_train_scaled = X_train_scaled[X_with_const.columns].copy()
X_train_scaled.info()
X_test_scaled = pd.DataFrame(X_test_scaled,columns=X_test.columns, index=X_test.index).copy()
X_test_scaled = X_test_scaled[X_with_const.columns].copy()
X_test_scaled.info()


# #### Logistic Regssion running the model with reduced features and ploting the roc cuve for threshold

# In[75]:


log_reg = LogisticRegression(random_state=123, max_iter = 1000)
log_reg.fit(X_train_scaled, y_train)

y_train_pred_log_reg = log_reg.predict(X_train_scaled)
y_train_pred_log_reg_prob = log_reg.predict_proba(X_train_scaled)[:, 1]
y_test_pred_log_reg = log_reg.predict(X_test_scaled)
y_test_pred_log_reg_prob = log_reg.predict_proba(X_test_scaled)[:, 1]


# In[76]:


cm_log_reg_train = confusion_matrix(y_train, y_train_pred_log_reg)
cm_log_reg_test = confusion_matrix(y_test, y_test_pred_log_reg)

print('Logistic Regression Train Results')
disp_cm_log_reg_train = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg_train)
disp_cm_log_reg_train.plot()
print(classification_report(y_train,y_train_pred_log_reg))


# In[77]:


print('Logistic Regression Test Results')
cm_log_reg_test = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg_train)
cm_log_reg_test.plot()
print(classification_report(y_test,y_test_pred_log_reg))


# #### Ploting ROC Curve

# In[78]:


fpr, tpr, thresholds = roc_curve(y_train, y_train_pred_log_reg_prob)
roc_auc = roc_auc_score(y_train, y_train_pred_log_reg_prob)

# Plot the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('./images/ROC_Curve.svg')
plt.show()

# Calculate Youden's J statistic
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold}")


# #### Getting the Target as per the threshold

# In[79]:


y_pred_optimal = (y_train_pred_log_reg_prob >= optimal_threshold).astype(int)


# In[80]:


y_pred_test_optimal = (y_test_pred_log_reg_prob >= optimal_threshold).astype(int)


# In[81]:


cm_log_reg_train = confusion_matrix(y_train, y_pred_optimal)
cm_log_reg_test = confusion_matrix(y_test, y_pred_test_optimal)


# In[82]:


print('Logistic Regression Train Results as of Threshold')
cm_log_reg_train = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg_train)
cm_log_reg_train.plot()
print(classification_report(y_train,y_pred_optimal))


# In[83]:


print('Logistic Regression Test Results as of Threshold')
ConfusionMatrixDisplay(confusion_matrix=cm_log_reg_test).plot()
print(classification_report(y_train,y_pred_optimal))


# #### Still The Logistic Model is not performing good based on the recall which is only in the 30%

# #### Hypertuning for Random Forest

# In[84]:


# Commeting this cell not to run
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],           # Number of trees
    'max_depth': [None, 10, 20, 30],          # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],          # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],            # Minimum samples required in a leaf node
    'max_features': ['sqrt', 'log2', None]    # Number of features to consider when looking for the best split
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='recall',  # Metric to optimize
    cv=5,                # 5-fold cross-validation
    verbose=2,           # Display progress
    n_jobs=-1            # Use all available cores
)

#grid_search.fit(X_train_scaled, y_train)

# Best hyperparameters and corresponding score
#best_params = grid_search.best_params_
#best_score = grid_search.best_score_

# Display the results
#print("Best Parameters:")
#print(best_params)
#print("\nBest Recall:")
#print(best_score)


# #### Best Parameters: {'max_depth': None, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200}
# Best Recall: 0.07703225806451612

# #### Applying the parameters to the Random forest model

# In[85]:


rf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth= None, max_features= None, min_samples_leaf= 4, min_samples_split=10 )
rf.fit(X_train_scaled, y_train)
y_train_pred_rf = rf.predict(X_train_scaled)
y_train_pred_rf_prob = rf.predict_proba(X_train_scaled)[:, 1]
y_test_pred_rf = rf.predict(X_test_scaled)
y_test_pred_rf_prob = rf.predict_proba(X_test_scaled)[:, 1]


# In[86]:


cm_rf_train = confusion_matrix(y_train, y_train_pred_rf)
cm_rf_test = confusion_matrix(y_test, y_test_pred_rf)


# In[87]:


print('Random Forest Train Results')
disp_cm_rf_train = ConfusionMatrixDisplay(confusion_matrix=cm_rf_train)
disp_cm_rf_train.plot()
print(classification_report(y_train,y_train_pred_rf))


# In[88]:


print('Random Forest Test Results')
disp_cm_rf_train = ConfusionMatrixDisplay(confusion_matrix=cm_rf_test)
disp_cm_rf_train.plot()
print(classification_report(y_test,y_test_pred_rf))


# #### Using SMOTE to Balance the split of the defaults in the data

# In[89]:


# recreate the data
#Create the Default Column 
funda_data_imputted['Defaulter'] = np.where(funda_data_imputted['Networth Next Year'] <= 0, 1, 0)
#check the % split of the defaulters
funda_data_imputted['Defaulter'].value_counts(normalize=True)*100
#spilt data in 70 30 ratio of train and test 
X = funda_data_imputted.drop(['Defaulter','Networth Next Year'], axis=1).copy()
y = funda_data_imputted['Defaulter'].copy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=123,stratify=funda_data_imputted['Defaulter'])


# In[90]:


# Initialize SMOTE
smote = SMOTE(sampling_strategy=0.75,random_state=42)

# Apply SMOTE to the training set
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# In[91]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)


# In[92]:


X_train_scaled = pd.DataFrame(X_train_scaled,columns=X_train_resampled.columns, index=X_train_resampled.index).copy()
X_train_scaled = X_train_scaled[X_with_const.columns].copy()
X_train_scaled.info()
X_test_scaled = pd.DataFrame(X_test_scaled,columns=X_test.columns, index=X_test.index).copy()
X_test_scaled = X_test_scaled[X_with_const.columns].copy()
X_test_scaled.info()


# #### Logistic regression Post SMOTE

# In[93]:


log_reg = LogisticRegression(random_state=123, max_iter = 1000)
log_reg.fit(X_train_scaled, y_train_resampled)

y_train_pred_log_reg = log_reg.predict(X_train_scaled)
y_train_pred_log_reg_prob = log_reg.predict_proba(X_train_scaled)[:, 1]
y_test_pred_log_reg = log_reg.predict(X_test_scaled)
y_test_pred_log_reg_prob = log_reg.predict_proba(X_test_scaled)[:, 1]


# In[94]:


cm_log_reg_train = confusion_matrix(y_train_resampled, y_train_pred_log_reg)
cm_log_reg_test = confusion_matrix(y_test, y_test_pred_log_reg)


# In[95]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
print('Logistic Regression Train Results as of Threshold')
cm_log_reg_train = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg_train)
cm_log_reg_train.plot()
print(classification_report(y_train_resampled,y_train_pred_log_reg))


# In[96]:


print('Logistic Regression Test Results as of Threshold')
ConfusionMatrixDisplay(confusion_matrix=cm_log_reg_test).plot()
print(classification_report(y_test,y_test_pred_log_reg))


# #### Random Forest Post SMOTE

# In[97]:


rf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth= None, max_features= None, min_samples_leaf= 4, min_samples_split=10 )
rf.fit(X_train_scaled, y_train_resampled)
y_train_pred_rf = rf.predict(X_train_scaled)
y_train_pred_rf_prob = rf.predict_proba(X_train_scaled)[:, 1]
y_test_pred_rf = rf.predict(X_test_scaled)
y_test_pred_rf_prob = rf.predict_proba(X_test_scaled)[:, 1]


# In[98]:


cm_rf_train = confusion_matrix(y_train_resampled, y_train_pred_rf)
cm_rf_test = confusion_matrix(y_test, y_test_pred_rf)


# In[99]:


print('Random Forest Train Results')
disp_cm_rf_train = ConfusionMatrixDisplay(confusion_matrix=cm_rf_train)
disp_cm_rf_train.plot()
print(classification_report(y_train_resampled,y_train_pred_rf))


# In[100]:


print('Random Forest Test Results')
disp_cm_rf_train = ConfusionMatrixDisplay(confusion_matrix=cm_rf_test)
disp_cm_rf_train.plot()
print(classification_report(y_test,y_test_pred_rf))


# ## Question 5

# ### Model Performance Comparison

# #### Logistic regression and Random Forest are not the most suitable models for this classification problem
# - We did get better and between in each step
#     - From base to VIF and P value Treatment to SMOTE
# - We would need to look at other methods like PCA or other modell like boosting/bagging to see if we can get a better model
# - The final recall on Test by the models were  0.25 for LR and 0.22 for Random Forest which was much better than the 0.01 for LR adn 0.10 for RF what we started with, but both these two models will not work in this case.
# - important Feature list post VIF and Pvalue elimination are ordered as:
#     - Raw material turnover              0.022
#     - Total expenses                     0.020
#     - Current ratio (times)              0.012
#     - Cash profit as % of total income   0.005
#     - Reserves and funds                 0.000 

# ## Question 6

# ### Actionable Insights & Recommendations
# #### We would need to look at other methods like PCA or other modell like boosting/bagging to see if we can get a better model

# # Part B

# ## Question 1

# ### Draw a Stock Price Graph (Stock Price vs Time) for the given stocks - Write observations

# In[101]:


stock_data = pd.read_csv('./data/Market_Risk_Data_coded.csv')
stock_data.head()


# In[102]:


stock_data.isnull().sum()


# In[103]:


# Convert the 'Date' column to datetime
stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%d-%m-%Y')


# In[104]:


# Plot stock prices over time
plt.figure(figsize=(12, 6))
for column in stock_data.columns[1:]:
    plt.plot(stock_data['Date'], stock_data[column], label=column)

plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid()
plt.savefig('./images/Stock_Price_Trend.svg')
plt.show()


# ### Observations
#     - The graph shows the stock price trends for all five stocks over the given period.
#     - Some stocks show consistent growth, while others exhibit high volatility or downward trends.
#     - Bharti Airtel has the highest gains vs Yes Bank who has the least

# ## Question 2

# ### Stock Returns Calculation and Analysis

# ### Calculate Returns for all stocks

# In[105]:


# Calculate weekly returns for each stock
returns = stock_data.set_index('Date').pct_change() * 100  # Returns in percentage


# ### Calculate the Mean and Standard Deviation for the returns of all stocks

# In[106]:


mean_returns = returns.mean()
std_dev_returns = returns.std()


# ### Draw a plot of Mean vs Standard Deviation for all stock returns

# In[107]:


# Create a plot of Mean vs Standard Deviation for stock returns
plt.figure(figsize=(8, 6))
plt.scatter(std_dev_returns, mean_returns, color='b', s=100, alpha=0.7)
for i, stock in enumerate(mean_returns.index):
    plt.text(std_dev_returns[i], mean_returns[i], stock, fontsize=9, ha='right')

plt.title('Mean vs Standard Deviation of Stock Returns')
plt.xlabel('Standard Deviation of Returns')
plt.ylabel('Mean Returns')
plt.grid()
plt.savefig('./images/Stock_Returns_Mean_vs_STD.svg')
plt.show()


# ### Observations and Inferences
#     - The mean returns indicate the average performance of each stock.
#     - The standard deviation of returns shows the volatility (risk) associated with each stock.
#     - The Mean vs Standard Deviation plot highlights which stocks offer higher returns for a given level of risk.
#     - Bharti Airtel and ITC Limited give mid range returns for the least volatity, DLF and Tata have higher returns than them but have mid range volatility, Yes Bank has the least returns and the most volatility

# ## Question 3

# ### Actionable insights and recommendations
#     - High-return stocks with low volatility are ideal for risk-averse investors.(Airtel/ITC)
#     - Stocks with high volatility but significant returns might be suitable for risk-tolerant investors.(DLF/Tata)
#     - Portfolio diversification can balance high-return and low-risk stocks to achieve an optimal risk-adjusted return.(Mix of the above)
