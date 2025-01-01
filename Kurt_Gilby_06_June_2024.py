#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_excel("./data/compactiv.xlsx")


# In[3]:


data.head()


# # Problem 1

# ## Question 1 : Define the problem and perform exploratory Data Analysis
# ### Problem Definition :
# We are given data from a workstation that is operating in a multi-user university department, the data has activity measures of the computers systems, Users use the workstation for tasks like internet access, file editing and CPU-intensive Programs, using the data set given with the system measures, we need to find a linear equation which can best predict the "Proportion of time that the CPU runs in user mode, given the input parameters that are made up of the activites listed in the data.
# 
# To do this we will apply linear regression models to the data set to find the best fit line which gives us the linear equation of the independant attribes to the dependant attribute which in this case is "usr".
# 
# Definition of the Data given is as follows:
# 
# | Attribute | Description |
# |-----------|-------------|
# | lread | Reads (transfers per second ) between system memory and user memory |
# | lwrite | writes (transfers per second) between system memory and user memory |
# | scall | Number of system calls of all types per second |
# | sread | Number of system read calls per second . |
# | swrite | Number of system write calls per second . |
# | fork | Number of system fork calls per second. |
# | exec | Number of system exec calls per second. |
# | rchar | Number of characters transferred per second by system read calls |
# | wchar | Number of characters transfreed per second by system write calls |
# | pgout | Number of page out requests per second |
# | ppgout | Number of pages, paged out per second |
# | pgfree | Number of pages per second placed on the free list. |
# | pgscan | Number of pages checked if they can be freed per second |
# | atch | Number of page attaches (satisfying a page fault by reclaiming a page in memory) per second |
# | pgin | Number of page-in requests per second |
# | ppgin | Number of pages paged in per second |
# | pflt | Number of page faults caused by protection errors (copy-on-writes). |
# | vflt | Number of page faults caused by address translation . |
# | runqsz | Process run queue size (The number of kernel threads in memory that are waiting for a CPU to run.Typically, this value should be less than 2. Consistently higher values mean that the system might be CPU-bound.) |
# | freemem | Number of memory pages available to user processes |
# | freeswap | Number of disk blocks available for page swapping. |
# | usr | Portion of time (%) that cpus run in user mode |

# ### Check Shape, Data Types, Statistical Summary :
# - The Data has 8192 observations and 21 independant features/attributes and 1 dependant attribute
# - The Data has 13 number of features with data type : float64.
# - The Data has 8 number of features with data type : int64.
# - The Data has 1 number of features with data type : object.
# - "runqsz" is of data type "object", we need to check this for values counts and convert to int or float.
# - "Null" values are seen in "rchar" and "wchar", we need to check this and treat accordingly
# - Statistical Summary:
# | Attribute | Statistics | Observations |
# | --- | --- | --- |
# | lread | count: 8192 , mean: 19.56, stdev: 53.354,min: 0, 25th_Per: 2.0, 50th_Per: 7.0, 75th_Per: 20.0, max: 1845 | no missing values-potiential outliers-high variance|
# | lwrite | count: 8192 , mean: 13.106, stdev: 29.892,min: 0, 25th_Per: 0.0, 50th_Per: 1.0, 75th_Per: 10.0, max: 575 | no missing values-potiential outliers-high variance-25th Percentile have 0 value.|
# | scall | count: 8192 , mean: 2306.318, stdev: 1633.617,min: 109, 25th_Per: 1012.0, 50th_Per: 2051.5, 75th_Per: 3317.25, max: 12493 | no missing values-potiential outliers-high variance.|
# | sread | count: 8192 , mean: 210.48, stdev: 198.98,min: 6, 25th_Per: 86.0, 50th_Per: 166.0, 75th_Per: 279.0, max: 5318 |  no missing values-potiential outliers-high variance.|
# | swrite | count: 8192 , mean: 150.058, stdev: 160.479,min: 7, 25th_Per: 63.0, 50th_Per: 117.0, 75th_Per: 185.0, max: 5456 | no missing values-potiential outliers-high variance.|
# | fork | count: 8192 , mean: 1.885, stdev: 2.479,min: 0.0, 25th_Per: 0.4, 50th_Per: 0.8, 75th_Per: 2.2, max: 20.12 | no missing values-potiential outliers-high variance.|
# | exec | count: 8192 , mean: 2.792, stdev: 5.212,min: 0.0, 25th_Per: 0.2, 50th_Per: 1.2, 75th_Per: 2.8, max: 59.56 | no missing values-potiential outliers-high variance.|
# | rchar | count: 8088 , mean: 197385.728, stdev: 239837.494,min: 278.0, 25th_Per: 34091.5, 50th_Per: 125473.5, 75th_Per: 267828.75, max: 2526649.0 | missing values-potiential outliers-high variance.|
# | wchar | count: 8177 , mean: 95902.993, stdev: 140841.708,min: 1498.0, 25th_Per: 22916.0, 50th_Per: 46619.0, 75th_Per: 106101.0, max: 1801623.0 | missing values-potiential outliers-low variance.|
# | pgout | count: 8192 , mean: 2.285, stdev: 5.307,min: 0.0, 25th_Per: 0.0, 50th_Per: 0.0, 75th_Per: 2.4, max: 81.44 | no missing values-potiential outliers-high variance-50 percent records with zero|
# | ppgout | count: 8192 , mean: 5.977, stdev: 15.215,min: 0.0, 25th_Per: 0.0, 50th_Per: 0.0, 75th_Per: 4.2, max: 184.2 | no missing values-potiential outliers-high variance-50 percent records with zero|
# | pgfree | count: 8192 , mean: 11.92, stdev: 32.364,min: 0.0, 25th_Per: 0.0, 50th_Per: 0.0, 75th_Per: 5.0, max: 523.0 | no missing values-potiential outliers-high variance-50 percent records with zero|
# | pgscan | count: 8192 , mean: 21.527, stdev: 71.141,min: 0.0, 25th_Per: 0.0, 50th_Per: 0.0, 75th_Per: 0.0, max: 1237.0 | no missing values-potiential outliers-high variance-75 percent records with zero|
# | atch | count: 8192 , mean: 1.128, stdev: 5.708,min: 0.0, 25th_Per: 0.0, 50th_Per: 0.0, 75th_Per: 0.6, max: 211.58 | no missing values-potiential outliers-high variance-50 percent records with zero|
# | pgin | count: 8192 , mean: 8.278, stdev: 13.875,min: 0.0, 25th_Per: 0.6, 50th_Per: 2.8, 75th_Per: 9.765, max: 141.2 |no missing values-potiential outliers-high variance|
# | ppgin | count: 8192 , mean: 12.389, stdev: 22.281,min: 0.0, 25th_Per: 0.6, 50th_Per: 3.8, 75th_Per: 13.8, max: 292.61 |no missing values-potiential outliers-high variance|
# | pflt | count: 8192 , mean: 109.794, stdev: 114.419,min: 0.0, 25th_Per: 25.0, 50th_Per: 63.8, 75th_Per: 159.6, max: 899.8 |no missing values-potiential outliers-high variance|
# | vflt | count: 8192 , mean: 185.316, stdev: 191.001,min: 0.2, 25th_Per: 45.4, 50th_Per: 120.4, 75th_Per: 251.8, max: 1365.0 |no missing values-potiential outliers-high variance|
# | freemem | count: 8192 , mean: 1763.456, stdev: 2482.105,min: 55, 25th_Per: 231.0, 50th_Per: 579.0, 75th_Per: 2002.25, max: 12027 |no missing values-potiential outliers-high variance|
# | freeswap | count: 8192 , mean: 1328125.96, stdev: 422019.427,min: 2, 25th_Per: 1042623.5, 50th_Per: 1289289.5, 75th_Per: 1730379.5, max: 2243187 |no missing values-low variance|
# | usr | count: 8192 , mean: 83.969, stdev: 18.402,min: 0, 25th_Per: 81.0, 50th_Per: 89.0, 75th_Per: 94.0, max: 99 |no missing values-low variance|

# In[4]:


data.shape
print(f'The Data has {data.shape[0]} observations and {data.shape[1]-1} independant features/attributes and 1 dependant attribute')


# In[5]:


data.info()
for idx, val in enumerate(data.dtypes.value_counts()):
    print(f'The Data has {val} number of features with data type : {data.dtypes.value_counts().index[idx]}.')


# In[6]:


data.describe().T.round(3)


# In[7]:


print('| Attribute | Statistics |')
print('| --- | --- |')
for col in data.columns[data.dtypes != 'object']:
    print(f'| {data[col].name} | count: {data[col].count()} , mean: {data[col].mean().round(3)}, stdev: {round(data[col].std(),3)},min: {round(data[col].min(),3)}, 25th_Per: {data[col].quantile(.25).round(3)}, 50th_Per: {data[col].quantile(.50).round(3)}, 75th_Per: {data[col].quantile(.75).round(3)}, max: {round(data[col].max(),3)} |')


# ### Univariate analysis:
# - plotting boxplots
#     - The features are on different scales, but the scale for rchar,wchar and freeswap is very different, plot these seperate than the others.
#     - Also the features scall,sread,swrite, pgfree, pgscan, pflt, vflt and freemem are on a medium scale, plot these seperate.
#     - all the features have outliers
#     - all the feature are skewed
# - plotting kde plots
#     - plotting the data collectively but due to the different scales its hard to compare
# - plotting the data seperately
#     - boxplots
#         - lread outliers are present, skewed to the right, High Variance.
#         - lwrite outliers are presen, skewed to the right, High Variance.
#         - scall outliers are present, skewed to the right, High Variance.
#         - sread outliers are present, skewed to the right, High Variance.
#         - swrite outliers are present, skewed to the right, High Variance.
#         - fork outliers are present, skewed to the right, High Variance.
#         - exec outliers are present, skewed to the right, High Variance.
#         - rchar outliers are present, skewed to the right, High Variance.
#         - wchar outliers are present, skewed to the right, High Variance.
#         - pgout outliers are present, skewed to the right, High Variance.
#         - ppgout outliers are present, skewed to the right, High Variance.
#         - pgfree outliers are present, skewed to the right, High Variance.
#         - pgscan outliers are present, skewed to the right, High Variance.
#         - atch outliers are present, skewed to the right, High Variance.
#         - pgin outliers are present, skewed to the right, High Variance.
#         - ppgin outliers are present, skewed to the right, High Variance.
#         - pflt outliers are present, skewed to the right, High Variance.
#         - vflt outliers are present, skewed to the right, High Variance.
#         - freemem outliers are present, skewed to the right, High Variance.
#         - freeswap outliers are present, skewed to the left, High Variance.
#         - usr outliers are present, skewed to the left, High Variance.
#   - kdeplots
#       - lread follows bellcure, alot of small values, small IQR, long right tail, confrims the above observations from box plots.
#       - lwrite follows bellcure, alot of small values, small IQR, long right tail, confrims the above observations from box plots.
#       - scall follows bellcure- two peaks seen, most values between 0 and 7500, spread IQR, long right tail, confrims the above observations from box plots.
#       - sread follows bellcure, alot of small values, small IQR, long right tail, confrims the above observations from box plots.
#       - swrite follows bellcure, alot of small values, small IQR, long right tail, confrims the above observations from box plots.
#       - fork follows bellcure, most values between 0 and 10, spread IQR, long right tail, confrims the above observations from box plots.
#       - exec follows bellcure-small bumps around values 20 and 35, alot of small values, small IQR, long right tail, confrims the above observations from box plots.
#       - rchar follows bellcure, most values between 0.0 and 1.0, spread IQR, long right tail, confrims the above observations from box plots.
#       - wchar follows bellcure, most values between 0.0 and 0.5, spread IQR, long right tail, confrims the above observations from box plots.
#       - pgout follows bellcure, alot of small values, small IQR, long right tail, confrims the above observations from box plots.
#       - ppgout follows bellcure, alot of small values, small IQR, long right tail, confrims the above observations from box plots.
#       - pgfree follows bellcure, alot of small values, small IQR, long right tail, confrims the above observations from box plots.
#       - pgscan follows bellcure, alot of small values, small IQR, long right tail, confrims the above observations from box plots.
#       - atch follows bellcure, alot of small values, small IQR, long right tail, confrims the above observations from box plots.
#       - pgin follows bellcure, most values between 0 and 50, spread IQR, long right tail, confrims the above observations from box plots.
#       - ppgin follows bellcure, most values between 0 and 50, spread IQR, long right tail, confrims the above observations from box plots.
#       - pflt follows bellcure, most values between 0 and 400, spread IQR, long right tail, confrims the above observations from box plots.
#       - vflt follows bellcure, most values between 0 and 750, spread IQR, long right tail, confrims the above observations from box plots.
#       - freemem follows bellcure-small bump around 7500, most values between 0 and 7500, spread IQR, long right tail, confrims the above observations from box plots.
#       - free swap follows bellcure-bumps/peaks around [0.0,1.0,2.0] , most values between 1.0 and 2,0, spread IQR, long left tail, confrims the above observations from box plots.
#       - usr follows bellcure-bumps/peaks around [0,100] , most values between 60 and 100, spread IQR, long left tail, confrims the above observations from box plots.
#   - distribution of runqsz:
#     - The propotion for obs for "Not-CPU-Bound" to "CPU-Bound" is ~53% to 47%.
# ### Multivariate analysis:
# - boxplots split on runqsz
#     - There are difference seen in the distribution between for "Not-CPU-Bound" vs "CPU-Bound" in the following features
#         - scall, sread, fork, exec, rchar, wchar, pgin, ppgin, pflt, vflt, freemem, freeswap and usr
# - kdeplots split on runqz
#     - There are significant difference seen in the distribution between for "Not-CPU-Bound" vs "CPU-Bound" in the following features
#         - scall, rchar, freemem, freeswap, usr
# - pairplots
#     - Reviewing the pair plot we see there are potiential correlations between
#         -  lread and lwrite
#         -  scall and sread and swrite
#         -  swrite and rchar and wchar
#         -  fork and exec and pflt and vflt
#         -  exec and pflt and vflt
#         -  rchar and wchar and atch
#         -  pgout and ppgout and pgfree
#         -  pgfree and pgscan
#         -  pgin and ppgin
#         -  usr and fork and plt and vflt
# - heatmap
#     - Reviewing the heatmap plot we see there strong correlations between
#         - lread and lwrite
#         - scall and sread and swrite and fork and pflt and vflt
#         - sread and swrite and rchar and pflt and vflt
#         - fork and scall and exec and pflt and vflt
#         - exec and fork pflt and vflt
#         - rchar and sread and wchar
#         - wchar and rchar
#         - pgout and ppgout and pgfree and pgscan
#         - ppgout and pgout and pgfree and pgscan and pgin and ppgin
#         - pgfree and pgout and ppgout and pgscan and pgin and ppgin
#         - pgscan and pgout and ppgout and pgfree and pgin and ppgin
#         - pgin and ppgout and pgfree and pgscan and ppgin
#         - ppgin and ppgout and pgfree and pgscan and pgin
#         - pflt and scall and sread and fork and exec and vflt
#         - vflt and scall and sread and fork and exec and pflt
#         - freemem and freeswap
#         - freeswap and freemem and usr
#         - usr and freeswap
# 
# ### Key Observations:
# - There are Null values that need to be treated
# - The Scaling of the Data is not consistant and we would need to do scaling.
# - The all of the numeric features have a bellcure distribution with outliers.
# - skewness is present in all the numerical freatures, we should try and reduce the skewness.
# - There are few that have multipeaks, this is due to distribution difference due to "Not-CPU-Bound" to "CPU-Bound" states in the "runqsz", we need to take runqsz when running the model.
# - There is high correlation in the dataset or .5 and above between multiple features, we would like to use tools like PCA to take care of this correlations issues
# - 

# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
data_numeric =  data[data.columns[data.dtypes != 'object']]


# In[9]:


data_numeric.shape


# In[10]:


plt.figure(figsize=(20,5))
sns.boxplot(data_numeric)
plt.title('Univariate Analysis-BoxPlots')
plt.xlabel('features')
plt.ylabel('Values')
plt.tight_layout()
plt.savefig('./images/boxplots_all.svg')
plt.show()


# In[11]:


data_large_scale = data_numeric[['rchar','wchar','freeswap']]
plt.figure(figsize=(20,5))
sns.boxplot(data_large_scale)
plt.title('Univariate Analysis-BoxPlots-Large_scale')
plt.xlabel('features')
plt.ylabel('Values')
plt.tight_layout()
plt.savefig('./images/boxplots_large_Scale.svg')
plt.show()


# In[12]:


data_med_scale = data_numeric[['scall','sread','swrite','pgfree', 'pgscan', 'pflt', 'vflt', 'freemem']]
plt.figure(figsize=(20,5))
sns.boxplot(data_med_scale)
plt.title('Univariate Analysis-BoxPlots-Mid_scale')
plt.xlabel('features')
plt.ylabel('Values')
plt.tight_layout()
plt.savefig('./images/boxplots_Med_Scale.svg')
plt.show()


# In[13]:


data_small_scale = data_numeric.drop(['rchar','wchar','freeswap','scall','sread','swrite','pgfree', 'pgscan', 'pflt', 'vflt', 'freemem'],axis=1)
plt.figure(figsize=(20,5))
sns.boxplot(data_small_scale)
plt.title('Univariate Analysis-BoxPlots-Small_scale')
plt.xlabel('features')
plt.ylabel('Values')
plt.tight_layout()
plt.savefig('./images/boxplots_small_Scale.svg')
plt.show()


# In[14]:


plt.figure(figsize=(20,5))
sns.kdeplot(data_large_scale)
plt.title('Univariate Analysis-KDE Plot-large_scale')
plt.xlabel('Values')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('./images/boxplots_Large_Scale.svg')
plt.show()

plt.figure(figsize=(20,5))
sns.kdeplot(data_med_scale)
plt.title('Univariate Analysis-KDE Plot-large_scale')
plt.xlabel('Values')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('./images/boxplots_Large_Scale.svg')
plt.show()

plt.figure(figsize=(20,5))
sns.kdeplot(data_small_scale)
plt.title('Univariate Analysis-KDE Plot-large_scale')
plt.xlabel('Values')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('./images/boxplots_Large_Scale.svg')
plt.show()


# In[15]:


import numpy as np
n = len(data_numeric.columns)
ncols = 5
nrows = n//nclos if n % ncols == 0 else n//ncols + 1

fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(20,4*nrows))
axs = np.ravel(axs)
for idx in range(nrows * ncols):
    ax = axs[idx]  # get the current axis
    if idx < n:
        sns.boxplot(data_numeric[data_numeric.columns[idx]], ax=ax)
    else:
        fig.delaxes(ax)
fig.suptitle('Univariate Analysis of features using Box-Plots')
plt.subplots_adjust(top=1)
plt.tight_layout() 
plt.subplots_adjust(top=0.95)
plt.savefig('./images/boxplots.svg')
plt.show()


# In[16]:


n = len(data_numeric.columns)
ncols = 5
nrows = n//nclos if n % ncols == 0 else n//ncols + 1

fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(20,4*nrows))
axs = np.ravel(axs)
for idx in range(nrows * ncols):
    ax = axs[idx]  # get the current axis
    if idx < n:
        sns.kdeplot(data_numeric[data_numeric.columns[idx]], ax=ax)
        ax.set_title(data_numeric.columns[idx])
    else:
        fig.delaxes(ax)
fig.suptitle('Univariate Analysis of features using KDE Plots')
plt.subplots_adjust(top=1)
plt.tight_layout() 
plt.subplots_adjust(top=0.95)
plt.savefig('./images/kdeplots.svg')
plt.show()


# In[17]:


ax = sns.barplot(data['runqsz'].value_counts(normalize=True))
for i in ax.containers:
    ax.bar_label(i,fmt='%.2f%%',labels=[f'{j.get_height()*100:.2f}%' for j in i])
    ax.set_title('Univariate Analysis of runqsz using bar plot for proportions of categories')
plt.tight_layout()
plt.savefig('./images/runqsz_propotion.svg')
plt.show()


# In[18]:


n = len(data_numeric.columns)
ncols = 5
nrows = n//nclos if n % ncols == 0 else n//ncols + 1

fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(20,4*nrows))
axs = np.ravel(axs)
for idx in range(nrows * ncols):
    ax = axs[idx]  # get the current axis
    if idx < n:
        sns.boxplot(data=data,y=data[data_numeric.columns[idx]],x=data['runqsz'], hue=data['runqsz'], ax=ax)
    else:
        fig.delaxes(ax)
fig.suptitle('Multivariate Analysis of features using Box-Plots-Split on runqsz')
plt.subplots_adjust(top=1)
plt.tight_layout() 
plt.subplots_adjust(top=0.95)
plt.savefig('./images/boxplots_multi.svg')
plt.show()


# In[19]:


n = len(data_numeric.columns)
ncols = 5
nrows = n//nclos if n % ncols == 0 else n//ncols + 1

fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(20,4*nrows))
axs = np.ravel(axs)
for idx in range(nrows * ncols):
    ax = axs[idx]  # get the current axis
    if idx < n:
        sns.kdeplot(data=data,x=data[data_numeric.columns[idx]], hue=data['runqsz'], ax=ax)
    else:
        fig.delaxes(ax)
fig.suptitle('Multivariate Analysis of features using KDE-Plots-Split on runqsz')
plt.subplots_adjust(top=1)
plt.tight_layout() 
plt.subplots_adjust(top=0.95)
plt.savefig('./images/kdeplots_multi.svg')
plt.show()


# In[ ]:


sns.pairplot(data_numeric)
plt.title('Multivariate Analysis of features using Pair Plots')
plt.tight_layout()
plt.savefig('./images/pairplots.jpg')
plt.show()


# In[ ]:


corr = data_numeric.corr().round(2)
matrix = np.triu(corr)
annot = True
plt.figure(figsize=(12,10))
sns.heatmap(corr,annot=annot,fmt='.1f')
plt.title('Heatmap of numeric features')
plt.tight_layout()
plt.savefig('./images/heatmap.svg')
plt.show()


# In[ ]:


data.head()


# ## Question 2 : Data Pre-processing
# ### Prepare the data for modeling:
# #### Missing Values Teatment:
# - Values are missing in "rchar" and "wchar", since these states theotically possible but highly unlikely we will replace these values with the minimum values in the the series.
# - Check for duplicates.
# #### Outlier Detection (treat, if needed):
# - Outlier values are present but these are values that genuinely captured fromt the system, we will not do any Outlier Treatments.
# 
# #### Encoding:
# - "runqsz" is Label encoded, and treated as a categorical variable,i.e, we will not do any scaling or Transformation with this column.
# -  This is Encoded as {'CPU_Bound': 1, 'Not_CPU_Bound': 0}
# 
# 
# #### Feature Engineering:
# - **These steps will be run post spilting the data into "Train" and "Test" this is to ensure that the Transformation, Scaling of the Train data is not influenced by the Test data.**
# - All the features are highly skewed, we check transformations to check if reduce alot of this skewness, exploring varrious transformations:
#     - The Boxcox transformation would be done to the following features
#         - lread, scall, sread, swrite, fork, exec, rchar, wchar, pgout, ppgout, pgfree, pgscan, atch, pflt, vflt, freemem and freeswap
#     - The Cube root transformation would be done to the following features
#         -  lwrite, pgin and ppgin
#     - We will not transform the target variable of "usr"
# - We will scale the data to get all the attribute to the same scale.
#     - Since all of the features follow a bell curve we will scale using zscore method.
# 
# #### Train-Test Split:
# - We do a split of the data 70-30 Train-Test.
# - Apply the transformation and scaling (described in Feature Engineering) to the Train and Test post the Split.
# - Post the application using the heat map on the Train data we see:
#     - alot features have values greater than and equal to 0.5 and less than and equal to -0.5
#     - this indicates that moderate and strong correlations between features
#  

# In[ ]:


data.isnull().sum()


# In[ ]:


data['rchar'].describe().round(2)


# In[ ]:


data_copy = data.copy()


# In[ ]:


data_copy['rchar'] = data_copy['rchar'].fillna(data_copy['rchar'].min())
data_copy['wchar'] = data_copy['wchar'].fillna(data_copy['rchar'].min())


# In[ ]:


data_copy.isnull().sum()


# In[ ]:


data_copy.duplicated().sum()


# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


# In[ ]:


data_copy['runqsz']


# In[ ]:


#le = LabelEncoder()
#data_copy['runqsz'] = le.fit_transform(data_copy['runqsz'])
data_copy['runqsz'] = data_copy['runqsz'].map({'CPU_Bound': 1, 'Not_CPU_Bound': 0})
data_copy['runqsz'] = data_copy['runqsz'].astype('category')


# In[ ]:


data_copy['runqsz']


# In[ ]:


data_copy_numeric = data_copy[data_numeric.columns]
data_copy_numeric.head()
data_copy_numeric.describe().round().T
data_copy_numeric.isna().sum()


# In[ ]:


skew = []
for col in data_copy_numeric.columns:
    sk = data_copy_numeric[col].skew().round()
    skew.append(sk)

for idx, val in enumerate(skew):
    if val == 0:
        print(f'Skew is 0 for {data_copy_numeric.columns[idx]}')

ax = sns.barplot(y=skew,x=data_copy_numeric.columns)
for i in ax.containers:
    ax.bar_label(i,)
ax.set_xticks(data_copy_numeric.columns)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Skewness")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig('./images/skewness.svg')
plt.show()


# In[ ]:


data_copy_numeric.min().min()
data_positive = data_copy_numeric + abs(data_copy_numeric.min().min()) +.0000000001 # adding a small + value to the zero so that log transform does not throw error as log0 is an exception


# In[ ]:


data_log_trasformed = data_positive.apply(np.log)
skew = []
for col in data_copy_numeric.columns:
    sk = data_log_trasformed[col].skew().round()
    skew.append(sk)
for idx, val in enumerate(skew):
    if val == 0:
        print(f'Skew is 0 for {data_copy_numeric.columns[idx]}')
ax = sns.barplot(y=skew,x=data_copy_numeric.columns)
for i in ax.containers:
    ax.bar_label(i,)
ax.set_xticks(data_copy_numeric.columns)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Skewness Log Transform")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig('./images/skewness_log_transform.svg')
plt.show()


# In[ ]:


data_sqrt_trasformed = data_positive.apply(np.sqrt)
skew = []
for col in data_copy_numeric.columns:
    sk = data_sqrt_trasformed[col].skew().round()
    skew.append(sk)

for idx, val in enumerate(skew):
    if val == 0:
        print(f'Skew is 0 for {data_copy_numeric.columns[idx]}')
        
ax = sns.barplot(y=skew,x=data_copy_numeric.columns)
for i in ax.containers:
    ax.bar_label(i,)
ax.set_xticks(data_copy_numeric.columns)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Skewness Sqrt Transform")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig('./images/skewness_Sqrt_transform.svg')
plt.show()


# In[ ]:


data_cbrt_trasformed = data_positive.apply(np.cbrt)
skew = []
for col in data_copy_numeric.columns:
    sk = data_cbrt_trasformed[col].skew().round()
    skew.append(sk)

for idx, val in enumerate(skew):
    if val == 0:
        print(f'Skew is 0 for {data_copy_numeric.columns[idx]}')
        
ax = sns.barplot(y=skew,x=data_copy_numeric.columns)
for i in ax.containers:
    ax.bar_label(i,)
ax.set_xticks(data_copy_numeric.columns)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Skewness Cbrt Transform")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig('./images/skewness_cbrt_transform.svg')
plt.show()


# In[ ]:


data_reciprocal_trasformed = data_positive.apply(lambda x: 1/x)
skew = []
for col in data_copy_numeric.columns:
    sk = data_reciprocal_trasformed[col].skew().round()
    skew.append(sk)

for idx, val in enumerate(skew):
    if val == 0:
        print(f'Skew is 0 for {data_copy_numeric.columns[idx]}')
        
ax = sns.barplot(y=skew,x=data_copy_numeric.columns)
for i in ax.containers:
    ax.bar_label(i,)
ax.set_xticks(data_copy_numeric.columns)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Skewness Reciprocal Transform")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig('./images/skewness_inverse_transform.svg')
plt.show()


# In[ ]:


from scipy.stats import boxcox


# In[ ]:


data_boxcox_trasformed = data_positive.apply(lambda x: pd.Series(boxcox(x)[0]))
skew = []
for col in data_copy_numeric.columns:
    sk = data_boxcox_trasformed[col].skew().round()
    skew.append(sk)

for idx, val in enumerate(skew):
    if val == 0:
        print(f'Skew is 0 for {data_copy_numeric.columns[idx]}')
        
ax = sns.barplot(y=skew,x=data_copy_numeric.columns)
for i in ax.containers:
    ax.bar_label(i,)
ax.set_xticks(data_copy_numeric.columns)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Skewness Boxcox Transform")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig('./images/skewness_boxcox_transform.svg')
plt.show()


# In[ ]:


# - lread, scall, sread, swrite, fork, exec, rchar, wchar, pgout, ppgout, pgfree, pgscan, atch, pflt, vflt, freemem and freeswap
#     - The Cube root transformation would be done to the following features
#         -  lwrite, pgin and ppgin

boxcox_cols = ['lread', 'scall', 'sread', 'swrite', 'fork', 'exec', 'rchar', 'wchar', 'pgout', 'ppgout', 'pgfree', 'pgscan', 'atch', 'pflt', 'vflt', 'freemem','freeswap']
cbrt_cols = ['lwrite', 'pgin', 'ppgin']


# In[ ]:


from sklearn.model_selection import train_test_split

X = data_copy.drop('usr', axis=1)
y = data_copy['usr']

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[ ]:


X_train_numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
X_train_numeric = X_train[X_train_numeric_cols]
X_test_numeric = X_test[X_train_numeric_cols]


# In[ ]:


X_train_numeric


# In[ ]:


X_test_numeric


# In[ ]:


# adding small values to make zero values positive as transformation like boxcox expects +ve values
X_train_numeric.min().min()
X_train_positive = X_train_numeric + abs(X_train_numeric.min().min()) +.0000000001 

X_test_numeric.min().min()
X_test_positive = X_test_numeric + abs(X_test_numeric.min().min()) +.0000000001


# In[ ]:





# In[ ]:


# applying transformations to Train and Test
boxcox_cols = ['lread', 'scall', 'sread', 'swrite', 'fork', 'exec', 'rchar', 'wchar', 'pgout', 'ppgout', 'pgfree', 'pgscan', 'atch', 'pflt', 'vflt', 'freemem','freeswap']
cbrt_cols = ['lwrite', 'pgin', 'ppgin']

X_train_positive_bc_cols = X_train_positive[boxcox_cols]
X_train_positive_cbrt_cols = X_train_positive[cbrt_cols]

X_test_positive_bc_cols = X_test_positive[boxcox_cols]
X_test_positive_cbrt_cols = X_test_positive[cbrt_cols]


# In[ ]:


X_test_positive_cbrt_cols


# In[ ]:


# Apply the Box-Cox transformation to each column of the training data
X_train_boxcox = X_train_positive_bc_cols.apply(lambda x:boxcox(x)[0])

# Get the lambdas
lambdas = X_train_positive_bc_cols.apply(lambda x: boxcox(x)[1])

# Apply same transformation on test
X_test_boxcox = X_test_positive_bc_cols.apply(lambda x: boxcox(x, lmbda=lambdas[x.name]))


# In[ ]:


X_test_boxcox


# In[ ]:


# Apply the Cbrt transformation to each column of the training data
X_train_cbrt = X_train_positive_cbrt_cols.apply(np.cbrt)

# Apply the Cbrt transformation to each column of the test data
X_test_cbrt = X_test_positive_cbrt_cols.apply(np.cbrt)


# In[ ]:


X_test_cbrt


# In[ ]:


X_train_transformed = pd.concat([X_train_boxcox,X_train_cbrt],axis=1)
X_train_transformed.head()


# In[ ]:


X_test_transformed = pd.concat([X_test_boxcox,X_test_cbrt],axis=1)
X_test_transformed.head()


# In[ ]:


# Scale Train and Test
scaler = StandardScaler()
# fit and transform train data
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_transformed), columns=X_train_transformed.columns,index=X_train_transformed.index)
# transform test data
X_test_scaled = pd.DataFrame(scaler.transform(X_test_transformed), columns=X_test_transformed.columns,index=X_test_transformed.index)


# In[ ]:


X_test_scaled


# In[ ]:


#check the correlation for the transformed and scaled values for training dataset
corr = X_train_scaled.corr().round(2)
matrix = np.triu(corr)
annot = True
plt.figure(figsize=(12,10))
sns.heatmap(corr,annot=annot,fmt='.1f')
plt.title('Training(Transformed and Scaled) Data set Heatmap')
plt.tight_layout()
plt.savefig('./images/heatmap_train_scaled_transformed.svg')
plt.show()


# In[ ]:


# adding the categorical column to the X_train_scaled and X_test_scaled
X_train_scaled = pd.concat([X_train_scaled,X_train['runqsz']],axis=1)
X_test_scaled = pd.concat([X_test_scaled,X_test['runqsz']],axis=1)


# In[ ]:


X_train_scaled.columns


# In[ ]:


X_test_scaled.columns


# ## Question 3 : Model Building - Linear regression
# ### Apply linear Regression using Sklearn:
# - We run LinearRegression using sklearn.linear_model
# - Fit the same to the X_train_scaled
# - running the model with all the features and using sklearn we get
#     - The MSE in training is ~124 and in test it is 133, there is more error in test.
#     - By the R2 we see that in the training the model explains ~62.7% of the variance in training and ~62.4% in testing.
# ### Using Statsmodels Perform checks for significant variables using the appropriate method:
# - We would check to run VIF as we know that there are multicolinearity so we would like to run this to get and remove features with Vif more than 5.
# - Using Statsmodeles we would like to run the model see what are the results and review the summary to get the p value of the features low p values means significant.
# - Results with all features:
#     - MSE: 124.11929490749988
#     - R-squared: 0.627402382682
# - ppgout has the highest vif (5951.473458) and high p>|t| value (0.808) dropping the same and rerunning, vif and Statsmodeles
# - pgout has the high vif (350.108379) and high p>|t| value (0.056) dropping the same and rerunning, vif and Statsmodeles
# - swrite has a high p>|t| value (0.280) dropping the same and rerunning, vif and Statsmodeles
# - pgin has a high vif (27) dropping the same and rerunning, vif and Statsmodeles
# - ppgin has a high p>|t| value (0.07) dropping the same and rerunning, vif and Statsmodeles
# - exec has a high p>|t| value (0.658 ) dropping the same and rerunning, vif and Statsmodeles
# - pgscan has a high p>|t| value (0.481 ) dropping the same and rerunning, vif and Statsmodeles
# - wchar has a high p>|t| value (0.157) dropping the same and rerunning, vif and Statsmodeles
# - Results removed features on the test data:
#     - Test MSE: 134.73827114064625
#     - Test R-squared: 0.6197317031796956
# - Even though the the "R-squared" decreases with the removal of the features, it is perffered as it removes multicolinearity and non-significant features 
# ### Create multiple models and check the performance of Predictions on Train and Test sets using Rsquare, RMSE & Adj Rsquare.
# - We would run the multiple models after droping the features due to multicolinearity and non-significance:
#     - ['ppgout','pgout','swrite','pgin','ppgin','exec','pgscan','wchar']
# - checking 2 models using Decision Tree Regressor and Linear Regression and comparing the RMSE to find the best model
#     - Decision Tree Regressor is performing better but overfitting, we will use cross validation to tune the hyperparameters.
# | Model | Train RMSE | Test RMSE | Training Score | Test Score |
# | --- | --- | --- | --- | --- |
# | Linear Regression | 11.207289| 11.607682| 0.621495| 0.619732 |
# | Decision Tree Regressor | 0.000000 | 3.884469 | 1.000000 |0.957414 |
#     - After pruning the Decision Tree Regresso we get a model which is working well on the Training and Test.
# | Model | Train RMSE | Test RMSE | Training Score | Test Score |
# | --- | --- | --- | --- | --- |
# | Linear Regression | 11.207289| 11.607682| 0.621495| 0.619732 |
# | Decision Tree Regressor | 2.952173 | 3.238304 | 0.973736  | 0.970404 |

# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


# Generate and fit the Model
lr = LinearRegression()

lr.fit(X_train_scaled,y_train)


# In[ ]:


# Generate the predictions for the dependant variable for train and test
y_train_pred = lr.predict(X_train_scaled)
X_test_scaled = X_test_scaled[X_train_scaled.columns]
y_test_pred = lr.predict(X_test_scaled)


# In[ ]:


print("Training MSE: ", mean_squared_error(y_train, y_train_pred))
print("Test MSE: ", mean_squared_error(y_test, y_test_pred))


# In[ ]:


print("Training R2 Score: ", r2_score(y_train, y_train_pred))
print("Test R2 Score: ", r2_score(y_test, y_test_pred))


# In[ ]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


vif = pd.DataFrame()
vif['vif_factor'] = [variance_inflation_factor(X_train_scaled.values,i) for i in range(X_train_scaled.shape[1])]
vif['features'] = X_train_scaled.columns
print(vif.sort_values(by='vif_factor', ascending=False))

# Add a constant to the independent value
X1 = sm.add_constant(X_train_scaled)

# Fit the model
model = sm.OLS(y_train, X1)
results = model.fit()

print(results.summary())

# Predict the values
y_pred = results.predict(X1)

# Calculate and print MSE and R-squared
mse = results.mse_resid
r2 = results.rsquared
print(f'MSE: {mse}')
print(f'R-squared: {r2}')


# In[ ]:


X_train_scaled_dropped_feature = X_train_scaled.drop('ppgout',axis=1)
vif = pd.DataFrame()
vif['vif_factor'] = [variance_inflation_factor(X_train_scaled_dropped_feature.values,i) for i in range(X_train_scaled_dropped_feature.shape[1])]
vif['features'] = X_train_scaled_dropped_feature.columns
print(vif.sort_values(by='vif_factor', ascending=False))

# Add a constant to the independent value
X1 = sm.add_constant(X_train_scaled_dropped_feature)

# Fit the model
model = sm.OLS(y_train, X1)
results = model.fit()

print(results.summary())

# Predict the values
y_pred = results.predict(X1)

# Calculate and print MSE and R-squared
mse = results.mse_resid
r2 = results.rsquared
print(f'MSE: {mse}')
print(f'R-squared: {r2}')


# In[ ]:


X_train_scaled_dropped_feature = X_train_scaled_dropped_feature.drop('pgout',axis=1)
vif = pd.DataFrame()
vif['vif_factor'] = [variance_inflation_factor(X_train_scaled_dropped_feature.values,i) for i in range(X_train_scaled_dropped_feature.shape[1])]
vif['features'] = X_train_scaled_dropped_feature.columns
print(vif.sort_values(by='vif_factor', ascending=False))

# Add a constant to the independent value
X1 = sm.add_constant(X_train_scaled_dropped_feature)

# Fit the model
model = sm.OLS(y_train, X1)
results = model.fit()

print(results.summary())

# Predict the values
y_pred = results.predict(X1)

# Calculate and print MSE and R-squared
mse = results.mse_resid
r2 = results.rsquared
print(f'MSE: {mse}')
print(f'R-squared: {r2}')


# In[ ]:


X_train_scaled_dropped_feature = X_train_scaled_dropped_feature.drop('swrite',axis=1)
vif = pd.DataFrame()
vif['vif_factor'] = [variance_inflation_factor(X_train_scaled_dropped_feature.values,i) for i in range(X_train_scaled_dropped_feature.shape[1])]
vif['features'] = X_train_scaled_dropped_feature.columns
print(vif.sort_values(by='vif_factor', ascending=False))

# Add a constant to the independent value
X1 = sm.add_constant(X_train_scaled_dropped_feature)

# Fit the model
model = sm.OLS(y_train, X1)
results = model.fit()

print(results.summary())

# Predict the values
y_pred = results.predict(X1)

# Calculate and print MSE and R-squared
mse = results.mse_resid
r2 = results.rsquared
print(f'MSE: {mse}')
print(f'R-squared: {r2}')


# In[ ]:


X_train_scaled_dropped_feature = X_train_scaled_dropped_feature.drop('pgin',axis=1)
vif = pd.DataFrame()
vif['vif_factor'] = [variance_inflation_factor(X_train_scaled_dropped_feature.values,i) for i in range(X_train_scaled_dropped_feature.shape[1])]
vif['features'] = X_train_scaled_dropped_feature.columns
print(vif.sort_values(by='vif_factor', ascending=False))

# Add a constant to the independent value
X1 = sm.add_constant(X_train_scaled_dropped_feature)

# Fit the model
model = sm.OLS(y_train, X1)
results = model.fit()

print(results.summary())

# Predict the values
y_pred = results.predict(X1)

# Calculate and print MSE and R-squared
mse = results.mse_resid
r2 = results.rsquared
print(f'MSE: {mse}')
print(f'R-squared: {r2}')


# In[ ]:


X_train_scaled_dropped_feature = X_train_scaled_dropped_feature.drop('ppgin',axis=1)
vif = pd.DataFrame()
vif['vif_factor'] = [variance_inflation_factor(X_train_scaled_dropped_feature.values,i) for i in range(X_train_scaled_dropped_feature.shape[1])]
vif['features'] = X_train_scaled_dropped_feature.columns
print(vif.sort_values(by='vif_factor', ascending=False))

# Add a constant to the independent value
X1 = sm.add_constant(X_train_scaled_dropped_feature)

# Fit the model
model = sm.OLS(y_train, X1)
results = model.fit()

print(results.summary())

# Predict the values
y_pred = results.predict(X1)

# Calculate and print MSE and R-squared
mse = results.mse_resid
r2 = results.rsquared
print(f'MSE: {mse}')
print(f'R-squared: {r2}')


# In[ ]:


X_train_scaled_dropped_feature = X_train_scaled_dropped_feature.drop('exec',axis=1)
vif = pd.DataFrame()
vif['vif_factor'] = [variance_inflation_factor(X_train_scaled_dropped_feature.values,i) for i in range(X_train_scaled_dropped_feature.shape[1])]
vif['features'] = X_train_scaled_dropped_feature.columns
print(vif.sort_values(by='vif_factor', ascending=False))

# Add a constant to the independent value
X1 = sm.add_constant(X_train_scaled_dropped_feature)

# Fit the model
model = sm.OLS(y_train, X1)
results = model.fit()

print(results.summary())

# Predict the values
y_pred = results.predict(X1)

# Calculate and print MSE and R-squared
mse = results.mse_resid
r2 = results.rsquared
print(f'MSE: {mse}')
print(f'R-squared: {r2}')


# In[ ]:


X_train_scaled_dropped_feature = X_train_scaled_dropped_feature.drop('pgscan',axis=1)
vif = pd.DataFrame()
vif['vif_factor'] = [variance_inflation_factor(X_train_scaled_dropped_feature.values,i) for i in range(X_train_scaled_dropped_feature.shape[1])]
vif['features'] = X_train_scaled_dropped_feature.columns
print(vif.sort_values(by='vif_factor', ascending=False))

# Add a constant to the independent value
X1 = sm.add_constant(X_train_scaled_dropped_feature)

# Fit the model
model = sm.OLS(y_train, X1)
results = model.fit()

print(results.summary())

# Predict the values
y_pred = results.predict(X1)

# Calculate and print MSE and R-squared
mse = results.mse_resid
r2 = results.rsquared
print(f'MSE: {mse}')
print(f'R-squared: {r2}')


# In[ ]:


X_train_scaled_dropped_feature = X_train_scaled_dropped_feature.drop('wchar',axis=1)
vif = pd.DataFrame()
vif['vif_factor'] = [variance_inflation_factor(X_train_scaled_dropped_feature.values,i) for i in range(X_train_scaled_dropped_feature.shape[1])]
vif['features'] = X_train_scaled_dropped_feature.columns
print(vif.sort_values(by='vif_factor', ascending=False))

# Add a constant to the independent value
X1 = sm.add_constant(X_train_scaled_dropped_feature)

# Fit the model
model = sm.OLS(y_train, X1)
results = model.fit()

print(results.summary())

# Predict the values
y_pred = results.predict(X1)

# Calculate and print MSE and R-squared
mse = results.mse_resid
r2 = results.rsquared
print(f'MSE: {mse}')
print(f'R-squared: {r2}')


# In[ ]:


X_test_scaled_dropped = X_test_scaled.drop(columns=['ppgout','pgout','swrite','pgin','ppgin','exec','pgscan','wchar'])
# Add a constant to the independent value
X1_test = sm.add_constant(X_test_scaled_dropped)

# Predict the values
y_test_pred = results.predict(X1_test[X1.columns])
# Calculate and print MSE and R-squared for test data
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
print(f'Test MSE: {mse}')
print(f'Test R-squared: {r2}')


# In[ ]:


get_ipython().system('pip install graphviz')


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
import graphviz


# In[ ]:


# Runing Linear regression and Decision Tree Regressor
# X_train and X_test removing the features that are non-significant and multicorelated which were identified above
X_train_scaled_dropped_feature
X_test_scaled_dropped

# Add a constant to the independent value
X1_train = sm.add_constant(X_train_scaled_dropped_feature)
X1_test = sm.add_constant(X_test_scaled_dropped[X_train_scaled_dropped_feature.columns])


dtr = DecisionTreeRegressor(random_state=42)
regression_model = LinearRegression()
ridge_model = Ridge()
lasso_model = Lasso()

models=[regression_model,ridge_model,lasso_model,dtr]

rmse_train=[]
rmse_test=[]
scores_train=[]
scores_test=[]

for i in models:  # Computation of RMSE and R2 values
    i.fit(X1_train,y_train)
    scores_train.append(i.score(X1_train, y_train))
    scores_test.append(i.score(X1_test, y_test))
    rmse_train.append(np.sqrt(mean_squared_error(y_train,i.predict(X1_train))))
    rmse_test.append(np.sqrt(mean_squared_error(y_test,i.predict(X1_test))))

print(pd.DataFrame({'Train RMSE': rmse_train,'Test RMSE': rmse_test,'Training Score':scores_train,'Test Score': scores_test},
            index=['Linear Regression','Ridge Linear Regression','Lasso Linear Regression','Decision Tree Regressor']))


# In[ ]:


# doing a grid search to find the optimum parameters for the Decision Tree Regressor
param_grid = {
    'max_depth': [10,15,20,25,30],
    'min_samples_leaf': [3, 15,30],
    'min_samples_split': [15,30,35,40,50],
}

dtr=DecisionTreeRegressor(random_state=42)

grid_search = GridSearchCV(estimator = dtr, param_grid = param_grid, cv = 5)
grid_search.fit(X1_train,y_train)

print(grid_search.best_params_)


# In[ ]:


dtr = DecisionTreeRegressor(max_depth=10,min_samples_split=15,min_samples_leaf=40,random_state=42)
regression_model = LinearRegression()
ridge_model = Ridge()
lasso_model = Lasso()

models=[regression_model,ridge_model,lasso_model,dtr]

rmse_train=[]
rmse_test=[]
scores_train=[]
scores_test=[]

for i in models:
    i.fit(X1_train,y_train)
    scores_train.append(i.score(X1_train, y_train))
    scores_test.append(i.score(X1_test, y_test))
    rmse_train.append(np.sqrt(mean_squared_error(y_train,i.predict(X1_train))))
    rmse_test.append(np.sqrt(mean_squared_error(y_test,i.predict(X1_test))))



print(pd.DataFrame({'Train RMSE': rmse_train,'Test RMSE': rmse_test,'Training Score':scores_train,'Test Score': scores_test},
            index=['Linear Regression','Ridge Linear Regression','Lasso Linear Regression','Decision Tree Regressor']))


# In[ ]:


# Visualize the decision tree
dot_data = export_graphviz(dtr, out_file='./images/decision_tree.dot', 
                           feature_names=X1_train.columns,  
                           filled=True, rounded=True,  
                           special_characters=True)


# In[ ]:


get_ipython().system('dot -Tpng ./images/decision_tree.dot -o decision_tree.png')


# In[ ]:


# Get the linear equation from the linear regression:
model = LinearRegression()
model.fit(X1_train,y_train)

coefficients = model.coef_
intercept = model.intercept_

# Print the linear equation
print("Linear Equation: y = ", end="")
for i in range(len(coefficients)):
    print(f"{coefficients[i].round(3)}x{X1_train.columns[i]} + ", end="")
print(intercept.round(3))

coff_dict = {}
for i in range(len(coefficients)):
    coff_dict[X1_train.columns[i]] = coefficients[i].round(3)

coff_dict_sorted = {k: v for k, v in sorted(coff_dict.items(), key=lambda item: abs(item[1]), reverse=True)}
print(coff_dict_sorted)   


# ## Question 4 : Business Insights & Recommendations
# ### Comment on the Linear Regression equation from the final model and impact of relevant variables (atleast 2) as per the equation:
# - Linear Equation: y = 0.0xconst + 0.674xlread + 1.86xscall + 1.052xsread + -5.876xfork + -2.281xrchar + 1.254xpgfree + 1.086xatch + -3.383xpflt + 4.882xvflt + 1.802xfreemem + 12.987xfreeswap + -1.004xlwrite + -6.91xrunqsz + 87.252
# - coefficients sorted absolute : {'freeswap': 12.987, 'runqsz': -6.91, 'fork': -5.876, 'vflt': 4.882, 'pflt': -3.383, 'rchar': -2.281, 'scall': 1.86, 'freemem': 1.802, 'pgfree': 1.254, 'atch': 1.086, 'sread': 1.052, 'lwrite': -1.004, 'lread': 0.674, 'const': 0.0}
# - **As per the equation freeswap has most impact on usr and runqsz has the most inverse impact on usr**
# - **As per the equation lread has least impact on usr and lwrite has the least inverse impact on usr**
# 
# ### Conclude with the key takeaways (actionable insights and recommendations) for the business:
# - if we run the "Linear Regression" in its current state it is not very usefull, in order to strengthing the model, we would like to perform tools Like PCA for futher feature engineering to try and get a more robust "Linear Regression"-
# - In the current state we do have the option of going with the "Decision Tree Regressor" which though may not have a linear relationship with the independant and dependant variables, still seems to a very good fit given the current training and test.
#     - This model however may need to be trained move frequently on live data and tested in real case senarios to make sure of its viability
# 

# # Problem 2
# ## Question 1 : Define the problem and perform exploratory Data Analysis
# ### Problem Definition :
# We are given the demographic and socio-economic survey data of married females who were either not pregnant or were uncertain of their pregnancy status during the survey, using logistic regression methods, we need to predict whether these women opt for a contraceptive method or not.
# 
# To do this we will apply logistic regression models to the data set to find the model which is a best classifies the women into the group of "Contraceptive method used": "No" or "Yes".
# we will use classification reports/summaries to identify the models that do the best job in accurately classifing these women.
# 
# Definition of the Data given is as follows:
# |Features| Description |
# | ---- | --- |
# | Wife_age | The age of the wife given as a number. |
# | Wife_education | The education of the wife given as a category with values for e.g as 'Primary', 'Secondary', 'Uneducated' etc.|
# | Husband_education |The education of the husband given as a category with values for e.g as 'Primary', 'Secondary', 'Uneducated' etc.|
# | No_of_children_born | The number of children born to the wife as a number.|
# | Wife_religion | The religon practiced by the wife as a category. |
# | Wife_Working | Is the wife working [No,Yes] |
# | Husband_Occupation | What is the category of the husbands occupation random values: [1,2,3,4] |
# | Standard_of_living_index | What is the category of standard of living values:['very low','low','high','very high'] |
# | Media_exposure | is the wife exposed to media values [Exposed, not Exposed] |
# | Contraceptive_method_used | does the wife use Contraceptive method values [No,Yes] |
# 
# ###  Check shape, Data types, statistical summary:
# - The data has 1473 observations, with 9 features and 1 dependant variable
# - There are missing values in "Wife_age" and "No_of_children_born"
# - There are 3 numerical features and 7 object features
# - statistical summary:
#     - Wife_age youngest = 16 years and oldest = 49 years, missing values are present
#     - Wife_education 4 unique values with "Tertiary" beign the most number of observations
#     - Husband_education 4 unique values with "Tertiary" beign the most number of observations
#     - No_of_children_born minimum = 1 and maximum = 16, missing values are present.
#     - Wife_religion 2 unique values with Scientology being the most number of observations
#     - Wife_Working 2 unique values with "No" being the most number of observations
#     - Husband_Occupation has 4 values, needs to be converted to categorical currently numeric
#     - Standard_of_living_index has 4 values with "Very High" being the most number of observations
#     - Media_exposure has 2 values with "Exposed" being the most number of observations
#     - Contraceptive_method_used has 2 values with "Yes" being the most number of observations
# ### Univariate analysis:
# - Husband_Occupation,Wife_education,Husband_education,Wife_religion,Wife_Working,Standard_of_living_index,Media_exposure,Contraceptive_method_used are categorical so will change these to category
# - Summary:
#     - Categorical features
#         - Husband_Occupation: The category "4" has only ~2% of the observations, the rest is close to a ~40%, ~30% and ~29% for 3,1 and 2 respectively.
#             - We may want to merge "4" with "3"
#         - Wife_education: Uneducated is only ~10% of the observations, , the rest is close to a ~40%, ~28% and ~23% for Tertiary, Secondary and Primary respectively.
#         - Husband_education: Uneducated is only ~3% of the observations, , the rest is close to a ~61%, ~24% and ~12% for Tertiary, Secondary and Primary respectively.
#         - Wife_religion: Non-Scientology is ~15% of the obersavations the rest is Scientology.
#         - Wife_working: Yes is ~25% of the obersavations the rest is No.
#         - Standard_of_living_index: Very Low is only ~9% of the observations, , the rest is close to a ~46%, ~29% and ~16% for Very High, High and Low respectively.
#         - Media_exposure: Not-Exposed is ~7% of the obersavations the rest is Exposed.
#         - Contraceptive_method_used: No is 43% of the obersavations the rest is Exposed.
#     - Numerical features
#         - Wife_age does not have any outliers
#         - No_of_children_born does not have any outliers
#         - We may want to change this to a categorical column using binning
# ### Multivariate analysis:
# - Summary:
#     - Categorical features
#         - Husband_Occupation: all of the categories are close to a 50-50 split vs Contraceptive_method_used for No-Yes, expect "1" which is close a 40-60 split, the mix of splits is different in the categories
#         - Wife_education: Secounday and Tertiary have a higher Yes Contraceptive_method_used vs Primary and Uneducated where No Contraceptive_method_used is higher
#         - Husband_education: Secounday and Tertiary have a higher Yes Contraceptive_method_used vs Primary and Uneducated where No Contraceptive_method_used is higher
#         - Wife_religion: Non-Scientology has a higher Yes Contraceptive_method_used vs Scientology
#         - Wife_working: No has a higher Yes Contraceptive_method_used vs Yes
#         - Standard_of_living_index: Low and Very Low have a lower Yes Contraceptive_method_used vs High and Very High
#         - Media_exposure: Exposed have a higher Yes Contraceptive_method_used vs Not-Exposed
#     - Numerical features:
#         - Wife_age: The median age of women of No Contraceptive_method_used is slightly higher than yes No Contraceptive_method_used
#             - Also there is more spread/variance in the ages of the women not using Contraceptive vs women who use
#             - would change this to categorical column using binning in data preprocessing
#         - No_of_children_born: The median No of childern born is lesser in No Contraceptive_method_used is slightly higher than yes No Contraceptive_method_used
#             - There are outliers in both the data sets
# ### Key meaningful observations on individual variables and the relationship between variables:
# -  Husband_Occupation Category "4" does not have enough observations and can be merged with "3".
# -  Husband_Occupation Category does not seen to have any influance on Contraceptive_method_used.
# -  The mix of education for both the husband and wife is as follows: Tertiary > Secoundary > Primary > Uneducated
# -  For both Wife_education and Husband_education as the education level increase the mix of Contraceptive_method_used Yes increases from ~30% on the Uneducated end to ~60% and above on the Tertiary end.
# -  Most of the dataset observations belong to the Scientology religion.(~85%)
# -  Wife_religion Non-Scientology women have a ~10% higher observation of Contraceptive_method_used Yes.
# -  Most of the dataset observations belong to the non-working women.(~75%)
# -  Wife_working does not seem to have to much of an impact on Contraceptive_method_used.
# -  The mix of Standard_of_living_index is as follows: Very High > High > Low > Very Low
# -   For Standard_of_living_index as the level increases the mix of Contraceptive_method_used Yes increases from ~38% on the Very Low end to ~64% on the Veery High end.
# -   Media_exposure does have a impact on Contraceptive_method_used.
# -  The data has a good mix of observations for Contraceptive_method_used, ~57% Yes vs ~43% No

# In[ ]:


contra_data = pd.read_excel("./data/Contraceptive_method_dataset.xlsx")
contra_data.head()


# In[ ]:


contra_data.shape


# In[ ]:


contra_data.info()


# In[ ]:


contra_data.describe(include='all').T


# In[ ]:


contra_data.rename(columns={'Wife_ education': 'Wife_education','Media_exposure ': 'Media_exposure'}, inplace=True)
cat_cols = ['Husband_Occupation','Wife_education','Husband_education','Wife_religion','Wife_Working','Standard_of_living_index','Media_exposure','Contraceptive_method_used']
for col in cat_cols:
    if col == 'Husband_Occupation':
        contra_data[col] = contra_data[col].astype('str')
    contra_data[col] = contra_data[col].astype('category')
contra_data.info()


# In[ ]:


contra_data_cat_cols = contra_data[cat_cols]

ncols = 5
nrows = np.ceil(len(contra_data_cat_cols.columns) / ncols).astype(int)
fig, axs = plt.subplots(nrows, ncols, figsize=(20, 4*nrows))
axs = axs.flatten()

for ax,col in zip(axs,contra_data_cat_cols.columns):
    summary = contra_data_cat_cols[col].value_counts(normalize=True)*100
    ax.bar(summary.index, summary.values)
    
    for i in range(len(summary)):
        ax.text(x=i, y=summary.iloc[i], s=f'{summary.iloc[i]:.2f}%', ha='center')
    ax.set_title(col)

for i in range(len(contra_data_cat_cols.columns), len(axs)):
    fig.delaxes(axs[i])

fig.suptitle('Univariate Analysis Proportion of Categorical Values')
plt.tight_layout()
plt.savefig('./images/propotion_plots.svg')
plt.show()


# In[ ]:


num_cols = [col for col in contra_data.columns if col not in cat_cols]
contra_data_num_cols = contra_data[num_cols]
ncols = 2
nrows = np.ceil(len(contra_data_num_cols.columns) / ncols).astype(int)
fig, axs = plt.subplots(nrows, ncols, figsize=(20, 4*nrows))
axs = axs.flatten()

for ax,col in zip(axs,contra_data_num_cols.columns):
    y = contra_data_num_cols[col]
    ax.boxplot(y.dropna().values)
    ax.set_title(col)

for i in range(len(contra_data_num_cols.columns), len(axs)):
    fig.delaxes(axs[i])

fig.suptitle('Univariate Analysis box plot Numerical Features')
plt.tight_layout()
plt.savefig('./images/contra_box_plots.svg')
plt.show()


# In[ ]:


contra_data_cat_cols = contra_data[cat_cols]

hue_col = 'Contraceptive_method_used'

ncols = 3
nrows = np.ceil(len(contra_data_cat_cols.columns) / ncols).astype(int)
fig, axs = plt.subplots(nrows, ncols, figsize=(20, 4*nrows))
axs = axs.flatten()

for ax,col in zip(axs,contra_data_cat_cols.columns):
    if col != 'Contraceptive_method_used' :
        # Calculate the proportions
        summary = contra_data_cat_cols.groupby(col)[hue_col].value_counts(normalize=True).mul(100)
        summary = summary.rename('proportion').reset_index()   
        # Create the bar plot
        bar_plot = sns.barplot(x=col, y='proportion', data=summary, ax=ax, hue=hue_col)
         
        # Add the values to the bars
        for p in bar_plot.patches:
            bar_plot.annotate(format(p.get_height(), '.2f') + '%', 
                              (p.get_x() + p.get_width() / 2., p.get_height()), 
                              ha = 'center', va = 'center', 
                              xytext = (0, 10), 
                              textcoords = 'offset points')
        
    
for i in range(len(contra_data_cat_cols.columns), len(axs)):
    fig.delaxes(axs[i])

fig.suptitle('Bivaritate Analysis Proportion of Categorical Values by Contraceptive_method_used')
plt.tight_layout()
plt.savefig('./images/Hue_propotion_plots.svg')
plt.show()


# In[ ]:


num_cols = [col for col in contra_data.columns if col not in cat_cols]
num_cols.append(hue_col)
contra_data_num_cols = contra_data[num_cols]

ncols = 2
nrows = np.ceil(len(contra_data_num_cols.columns) / ncols).astype(int)
fig, axs = plt.subplots(nrows, ncols, figsize=(20, 4*nrows))
axs = axs.flatten()

for ax,col in zip(axs,contra_data_num_cols.columns):
    if col != hue_col:
        sns.boxplot(x=hue_col, y=col, data=contra_data_num_cols, ax=ax)
        ax.set_title(col)

for i in range(len(contra_data_num_cols.columns), len(axs)):
    fig.delaxes(axs[i])

fig.suptitle('Bivariate Analysis box plot Numerical Features vs Contraceptive_method_used')
plt.tight_layout()
plt.savefig('./images/contra_hue_box_plots.svg')
plt.show()


# ## Question 2 : Data Pre-processing
# ### Prepare the data for modelling :
# ###  Missing value Treatment (if needed) 
# - Wife_age and No_of_children_born are treated for missing values with imputation of median value
# ### Outlier Detection(treat, if needed)
# - There are outliers in No_of_children_born but we will not treat these as these are real repersentations
# ### Feature Engineering (if needed) 
# - We will not engineer any features
# - we would scale the columns Wife_age and number of children.
# ### Encode the data
# - Encode Wife_education and Husband_education with values 0,1,2,3 for Uneducated, Primary, Secondary and Tertiary respectively.
# - Encode Wife_religon with values 0,1 for Non-Scientology and Scientology respectively.
# - Encode Wife_working with values 0,1 for No and Yes respectively.
# - Encode Standar_of_living_index with values 0,1,2,3,4 for Very Low, Low, High and Very High respectively
# - Encode Media_exposure with values 0,1 for Not-Exposed and Exposed respectively
# #### Train-Test Split:
# - We do a split of the data 70-30 Train-Test.
# - Apply scaling (described in Feature Engineering) to the Train and Test post the Split.

# In[ ]:


contra_data_copy = contra_data.copy()


# In[ ]:


contra_data_copy.isna().sum()


# In[ ]:


num_cols= ['Wife_age', 'No_of_children_born']


# In[ ]:


for col in num_cols:
    contra_data_copy[col] = contra_data_copy[col].fillna(contra_data_copy[col].median())


# In[ ]:


contra_data_copy.isna().sum()


# In[ ]:


contra_data_copy.describe(include='all').round().T


# In[ ]:


#Encoding the categorical columns
contra_data_copy.Husband_Occupation = contra_data_copy.Husband_Occupation.astype(int).astype('category')


# In[ ]:


# create encoding dictionaries
education_dict = {'Uneducated': 0,'Primary':1,'Secondary':2,'Tertiary':3}
religon_dict = {'Non-Scientology': 0,'Scientology':1}
working_dict = {'No': 0,'Yes':1}
soli_dict = {'Very Low': 0,'Low':1,'High':2,'Very High':3}
media_dict = {'Not-Exposed': 0,'Exposed':1}
contraceptive_dict = {'No': 0,'Yes':1}

contra_data_copy.Wife_education = contra_data_copy.Wife_education.replace(education_dict)
contra_data_copy.Husband_education = contra_data_copy.Husband_education.replace(education_dict)
contra_data_copy.Wife_religion = contra_data_copy.Wife_religion.replace(religon_dict)
contra_data_copy.Wife_Working = contra_data_copy.Wife_Working.replace(working_dict)
contra_data_copy.Standard_of_living_index = contra_data_copy.Standard_of_living_index.replace(soli_dict)
contra_data_copy.Media_exposure = contra_data_copy.Media_exposure.replace(media_dict)
contra_data_copy.Contraceptive_method_used = contra_data_copy.Contraceptive_method_used.replace(contraceptive_dict)


# In[ ]:


X = contra_data_copy.drop('Contraceptive_method_used', axis=1)
y = contra_data_copy['Contraceptive_method_used']

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[ ]:


X_train_scale = X_train[['Wife_age','No_of_children_born']]
X_test_scale = X_test[['Wife_age','No_of_children_born']]
X_train_scale.head()


# In[ ]:


scaler = StandardScaler()
# fit and transform train data
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_scale), columns=X_train_scale.columns,index=X_train_scale.index)
# transform test data
X_test_scaled = pd.DataFrame(scaler.transform(X_test_scale), columns=X_test_scale.columns,index=X_test_scale.index)


# In[ ]:


X_train.drop(columns=['Wife_age','No_of_children_born'],inplace=True)
X_train = pd.concat([X_train,X_train_scaled],axis=1)


# In[ ]:


X_test.drop(columns=['Wife_age','No_of_children_born'],inplace=True)
X_test = pd.concat([X_test,X_test_scaled],axis=1)


# In[ ]:


X_test


# ## Question 3 : Model Building and Compare the Performance of the Models :
# ### Build a Logistic Regression model - Build a Linear Discriminant Analysis model - Build a CART model 
# - We build Logistic Regression, LDA and a DecisionTree Classifier.
# - The results to compare is here:
# ![image.png](attachment:f50337f0-5fb6-4611-bbb7-3a9e5f0fd3d8.png)
# - f1 score which is the harmonic mean of the Recall and Precision is the highest for the "Logistic Regression", it also has decent recall but precision is a concern.
# - Precision score is good for the DecisionTree Classifier, depending on the use case we can got with either.
# ### Prune the CART model by finding the best hyperparameters using GridSearch:
# - optimum parameter for the CART Model is : {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 30, 'min_samples_split': 15}
# - we see that with this the CART Model is a good contender as the f1 score is the highest and the recall and precision scores is the highest
# - Runing the model with those parameter this is how the performance changes:
# - ![image.png](attachment:5c483236-3d77-47ee-a54c-9a8f5413c759.png)
# ### Compare the performance of all the models built and choose the best one with proper rationale
# - we see that with this the CART Model is a best as on the test data:
#     - f1 score is the highest
#     - the recall and precision scores is the highest
#     - the roc_auc_score is the highest
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# In[ ]:


dtr = DecisionTreeClassifier(random_state=42)
regression_model = LogisticRegression()
lda = LinearDiscriminantAnalysis()


models=[regression_model,dtr,lda]

accuracy_score_train=[]
accuracy_score_test=[]
precision_score_train=[]
precision_score_test=[]
recall_score_train=[]
recall_score_test=[]
f1_score_train=[]
f1_score_test=[]
roc_auc_score_train=[]
roc_auc_score_test=[]

for i in models:  # Computation of RMSE and R2 values
    i.fit(X_train,y_train)
    y_train_pred  = i.predict(X_train)
    y_test_pred = i.predict(X_test)
    
    accuracy_score_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_score_test.append(accuracy_score(y_test, y_test_pred))
    precision_score_train.append(precision_score(y_train, y_train_pred))
    precision_score_test.append(precision_score(y_test, y_test_pred))
    recall_score_train.append(recall_score(y_train, y_train_pred))
    recall_score_test.append(recall_score(y_test, y_test_pred))
    f1_score_train.append(f1_score(y_train, y_train_pred))
    f1_score_test.append(f1_score(y_test, y_test_pred))
    roc_auc_score_train.append(roc_auc_score(y_train, y_train_pred))
    roc_auc_score_test.append(roc_auc_score(y_test, y_test_pred))

print(pd.DataFrame({'Train_accuracy_score': accuracy_score_train,'Test_accuracy_score': accuracy_score_test,'Training_precision_score':precision_score_train,'Test_precision_score': precision_score_test, 'Training_recall_score':recall_score_train,'Test_recall_score': recall_score_test , 'Training_f1_score':f1_score_train,'Test_f1_score': f1_score_test, 'Training_roc_auc_score':roc_auc_score_train,'Test_roc_auc_score': roc_auc_score_test},
            index=['Logistic_Regression','Decision_Tree_Classifier', 'LDA_Regression']))


# In[ ]:


# using Grid search to Prune the DTC
param_grid = {
    'max_depth': [10,15,20,25,30],
    'min_samples_leaf': [3, 15,30],
    'min_samples_split': [15,30,35,40,50],
    'criterion' :['gini', 'entropy']
}

dtr=DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtr, param_grid = param_grid, cv = 3)


grid_search.fit(X_train,y_train)

print(grid_search.best_params_)


# In[ ]:


dtr = DecisionTreeClassifier(criterion = 'entropy', max_depth = 10, min_samples_leaf= 30, min_samples_split= 15,random_state=42)
regression_model = LogisticRegression()
lda = LinearDiscriminantAnalysis()


models=[regression_model,dtr,lda]

accuracy_score_train=[]
accuracy_score_test=[]
precision_score_train=[]
precision_score_test=[]
recall_score_train=[]
recall_score_test=[]
f1_score_train=[]
f1_score_test=[]
roc_auc_score_train=[]
roc_auc_score_test=[]

for i in models:  # Computation of RMSE and R2 values
    i.fit(X_train,y_train)
    y_train_pred  = i.predict(X_train)
    y_test_pred = i.predict(X_test)
    
    accuracy_score_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_score_test.append(accuracy_score(y_test, y_test_pred))
    precision_score_train.append(precision_score(y_train, y_train_pred))
    precision_score_test.append(precision_score(y_test, y_test_pred))
    recall_score_train.append(recall_score(y_train, y_train_pred))
    recall_score_test.append(recall_score(y_test, y_test_pred))
    f1_score_train.append(f1_score(y_train, y_train_pred))
    f1_score_test.append(f1_score(y_test, y_test_pred))
    roc_auc_score_train.append(roc_auc_score(y_train, y_train_pred))
    roc_auc_score_test.append(roc_auc_score(y_test, y_test_pred))

print(pd.DataFrame({'Train_accuracy_score': accuracy_score_train,'Test_accuracy_score': accuracy_score_test,'Training_precision_score':precision_score_train,'Test_precision_score': precision_score_test, 'Training_recall_score':recall_score_train,'Test_recall_score': recall_score_test , 'Training_f1_score':f1_score_train,'Test_f1_score': f1_score_test, 'Training_roc_auc_score':roc_auc_score_train,'Test_roc_auc_score': roc_auc_score_test},
            index=['Logistic_Regression','Decision_Tree_Classifier', 'LDA_Regression']))


# ## Question 4 : Business Insights & Recommendations
# 
# - if we run the "Logistic Regression" in its current state it is not very usefull, in order to strengthing the model, we would like to perform tools Like PCA for futher feature engineering to try and get a more robust "Logistic Regression"
# - We also need to check if we could get more data as the number of observations seems to be less it would be good to get more surveys done to get more info.
# - we could also explore further with one hot encoding techniques to see if any goodness is got with that models trained on that data.
# - we could explore tools and techniques like random forest, Support Vector, Kmeans clustering too to see if this helps identify patterns in the data.
# - In the current state we do have the option of going with the "Decision Tree Classifier" which though may not have a linear relationship with the independant and dependant variables, still seems to a very good fit given the current training and test.
# This model however may need to be trained move frequently on live data and tested in real case senarios to make sure of its viability

# In[ ]:




