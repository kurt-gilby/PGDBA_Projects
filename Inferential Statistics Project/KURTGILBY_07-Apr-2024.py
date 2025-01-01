#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import all libaries needed for this python notebook
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt 

import seaborn as sns


# # IS Project
# ## Problem 1
# ### A physiotherapist with a male football team is interested in studying the relationship between foot injuries and the positions at which the players play from the data collected.<br>
# <table border="1" width="657">
# <tbody>
# <tr>
# <td width="132">
# <p>&nbsp;</p>
# </td>
# <td width="51">
# <p>Striker</p>
# </td>
# <td width="64">
# <p>Forward</p>
# </td>
# <td width="140">
# <p>Attacking Midfielder</p>
# </td>
# <td width="58">
# <p>Winger</p>
# </td>
# <td width="64">
# <p><strong>Total</strong></p>
# </td>
# </tr>
# <tr>
# <td width="132">
# <p>Players Injured</p>
# </td>
# <td width="51">
# <p>45</p>
# </td>
# <td width="64">
# <p>56</p>
# </td>
# <td width="140">
# <p>24</p>
# </td>
# <td width="58">
# <p>20</p>
# </td>
# <td width="64">
# <p><strong>145</strong></p>
# </td>
# </tr>
# <tr>
# <td width="132">
# <p>Players Not Injured</p>
# </td>
# <td width="51">
# <p>32</p>
# </td>
# <td width="64">
# <p>38</p>
# </td>
# <td width="140">
# <p>11</p>
# </td>
# <td width="58">
# <p>9</p>
# </td>
# <td width="64">
# <p><strong>90</strong></p>
# </td>
# </tr>
# <tr>
# <td width="132">
# <p><strong>Total</strong></p>
# </td>
# <td width="51">
# <p><strong>77</strong></p>
# </td>
# <td width="64">
# <p><strong>94</strong></p>
# </td>
# <td width="140">
# <p><strong>35</strong></p>
# </td>
# <td width="58">
# <p><strong>29</strong></p>
# </td>
# <td width="64">
# <p><strong>235</strong></p>
# </td>
# </tr>
# </tbody>
# </table>

# ### Based on the above data, answer the following questions.

# In[2]:


# Create a dataframe to repersent the above data
df = pd.DataFrame({'Striker':[45,32],'Forward':[56,38],'Attacking Midfielder':[24,11],'Winger':[20,9]}, index = ['Players Injured','Players Not Injured'])
df.loc['Col_Total'] = df.sum(axis=0) #Calculation of the column totals
df.loc[:,'Row_Total'] = df.sum(axis=1) #Calculation of the row totals
df


# #### 1.1 What is the probability that a randomly chosen player would suffer an injury?

# In[3]:


# Porbability of randomly chosen player would suffer an injury = Number of players injured/ Total number of players
no_of_palyes_injured = df.loc['Players Injured']['Row_Total'] # Number of players injured
tot_no_of_players = df.loc['Col_Total']['Row_Total'] # Total number of players
prob_of_injury = no_of_palyes_injured/tot_no_of_players # Number of players injured/ Total number of players
prob_of_injury = round(prob_of_injury,4)
print(f'The probability that a randomly chosen player would suffer an injury is : {prob_of_injury} or {prob_of_injury*100}% of the time.')


# ##### Porbability of randomly chosen player would suffer an injury = **Number of players injured/ Total number of players**
# ##### From the table the "Number of players injured" = **145**
# ##### From the table the "Total Numebr players" = **235**
# ##### Number of players injured/ Total number of players = **145/235** = **0.617**
# ##### The probability that a randomly chosen player would suffer an injury is : **0.617 or 61.7%** of the time.

# #### 1.2 What is the probability that a player is a forward or a winger?

# In[4]:


# Porbability of randomly chosen player would would play in the position of Forward or Winger = Probability that randomly chosen player would would play in the position of Forward + Probability that randomly chosen player would would play in the position of Winger
# Probability that randomly chosen player would would play in the position of Forward = Number of palyers in Forward position / Total Number of Player
# Probability that randomly chosen player would would play in the position of Winger = Number of palyers in Winger position / Total Number of Player
# Porbability of randomly chosen player would would play in the position of Forward or Winger = (Number of palyers in Forward position + Number of palyers in Winger position)/Total Number of Player
tot_no_of_players # Total number of players as calculated in question 1.1
no_of_players_forward = df.loc['Col_Total']['Forward'] # Number of palyers in Forward position 
no_of_players_winger = df.loc['Col_Total']['Winger'] # Number of players in Winger position
prob_of_forward_or_winger = (no_of_players_forward + no_of_players_winger)/tot_no_of_players # (Number of palyers in Forward position + Number of palyers in Winger position)/Total Number of Player
prob_of_forward_or_winger = round(prob_of_forward_or_winger,4)
print(f'The probability that a randomly chosen player would be in a "Forward" or "Winger" position: {prob_of_forward_or_winger} or {round(prob_of_forward_or_winger*100,2)}% of the time.')


# ##### Porbability of randomly chosen player would would play in the position of Forward or Winger = **(Number of palyers in Forward position + Number of palyers in Winger position)/ Total number of players**
# ##### From the table the "Number of palyers in Forward position" = **94**
# ##### From the table the "Number of palyers in Winger position" = **29**
# ##### From the table the "Total Numebr players" = **235**
# ##### (Number of palyers in Forward position + Number of palyers in Winger position)/ Total number of players = **(94+29)/235** = **0.5234**
# ##### The probability that a randomly chosen player would be in a "Forward" or "Winger" position is : **0.5234 or 52.34%** of the time.

# #### 1.3 What is the probability that a randomly chosen player plays in a striker position and has a foot injury?

# In[5]:


# Probability that a randomly chosen player plays in a striker position and has a foot injury = Number of player in Striker position with injury/Total number of players
no_player_w_injury_and_striker_position = df.loc['Players Injured','Striker'] # Number of player in Striker position with injury
tot_no_strikers = df.loc['Col_Total','Striker'] # Total number of players as Strikers
prob_play_w_injury_and_striker_pos = no_player_w_injury_and_striker_position/tot_no_strikers # Number of player in Striker position with injury/Total number of players
prob_play_w_injury_and_striker_pos
prob_play_w_injury_and_striker_pos = round(prob_play_w_injury_and_striker_pos,4)
print(f'The probability that a randomly chosen player plays in a striker position and has a foot injury: {prob_play_w_injury_and_striker_pos} or {round(prob_play_w_injury_and_striker_pos*100,2)}% of the time.')


# ##### Probability that a randomly chosen player plays in a striker position and has a foot injury = **Number of player in Striker position with injury/Total number of players**
# ##### From the table the "Number of player in Striker position with injury" = **45**
# ##### From the table the "Total Numebr players" = **77**
# ##### Number of player in Striker position with injury/Total number of players = **45/77** = **0.5844**
# ##### The probability that a randomly chosen player plays in a striker position and has a foot injury is : **.5844 or 58.44%** of the time.

# #### 1.4 What is the probability that a randomly chosen injured player is a striker?

# In[6]:


# Probability that a randomly chosen injured player is a striker : Number of player who is a sticker and injured / Number of injurded players
no_player_w_injury_and_striker_position # Number of player in Striker position with injury from question 1.3
no_of_player_w_injury = df.loc['Players Injured']['Row_Total'] # Number of injurded players
prob_of_injured_player_is_striker = no_player_w_injury_and_striker_position/ no_of_player_w_injury
prob_of_injured_player_is_striker = round(prob_of_injured_player_is_striker,4)
print(f'The probability that a randomly chosen injured player is a striker: {prob_of_injured_player_is_striker} or {round(prob_of_injured_player_is_striker*100,2)}% of the time.')


# ##### Probability that a randomly chosen injured player is a striker : **Number of player who is a sticker and injured / Number of injurded players**
# ##### From the table the "Number of player who is a sticker and injured" = **45**
# ##### From the table the "Number of injurded players" = **145**
# ##### Number of player who is a sticker and injured / Number of injurded players = **45/145** = **0.3103**
# ##### The probability that a randomly chosen injured player is a striker is : **0.3103 or 31.03%** of the time.

# ## Problem 2
# ### The breaking strength of gunny bags used for packaging cement is normally distributed with a mean of 5 kg per sq. centimeter and a standard deviation of 1.5 kg per sq. centimeter. The quality team of the cement company wants to know the following about the packaging material to better understand wastage or pilferage within the supply chain. 
# ### Answer the questions below based on the given information.
# ### **(Provide an appropriate visual representation of your answers, without which marks will be deducted)**

# In[7]:


# Given:
# Breaking strength or gunny bags used for packaging cement is normally distributed.
# The population mean is 5 kg/cm^2
# The population standard deviation is 1.5 kg/cm^2
mu = 5 # in kg/cm^2 units
sigma = 1.5 # in kg/cm^2 units

# using the information given ploting the normal distribution curve
min_val = mu - 4*sigma # assuming a min val of 4 std dev below the mean
max_val = mu + 4*sigma # assuming a max val of 4 std dev above the mean
one_std_low = mu - sigma # get the lower bound of one std dev
one_std_high = mu + sigma # get the upper bound of one std dev
two_std_low = mu - 2*sigma # get the lower bound of two std dev
two_std_high = mu + 2*sigma # get the upper bound of two std dev
three_std_low = mu - 3*sigma # get the lower bound of three std dev
three_std_high = mu + 3*sigma # get the upper bound of three std dev
x = np.linspace(min_val,max_val,10000)
pdf = stats.norm.pdf(x,loc=mu,scale=sigma)
plt.figure(figsize=(10,6))
plt.title(f'Normal Distribution: mu:{mu} and sigma:{sigma}')
plt.plot(x,pdf)
plt.axvline(x=mu,color='b',linestyle='--',ymax=.95,ymin=0)
plt.axvline(x=one_std_low,color='g',linestyle='--',ymax=.58,ymin=0)
plt.axvline(x=one_std_high,color='g',linestyle='--',ymax=.58,ymin=0)
plt.axvline(x=two_std_low,color='y',linestyle='--',ymax=.16,ymin=0)
plt.axvline(x=two_std_high,color='y',linestyle='--',ymax=.16,ymin=0)
plt.axvline(x=three_std_low,color='r',linestyle='--',ymax=.05,ymin=0)
plt.axvline(x=three_std_high,color='r',linestyle='--',ymax=.05,ymin=0)
plt.ylabel('Probability Density')
plt.xlabel('Breaking Strength')
plt.legend(['pdf',f'mean: {mu}',f'sigma -: {one_std_low}',f'sigma +: {one_std_high}', f'2 sigma -: {two_std_low}',f'2 sigma +: {two_std_high}', f'3 sigma -: {three_std_low}', f'3 sigma +: {three_std_high}'])
plt.savefig("./images/norm_dist_mu_5_sigma_1_5.svg")
plt.show()


# #### Above is the plot of the Normal Distribution of the breaking strength of gunny bags used for packaging cement follows.

# In[8]:


#Declaring the given parameters in the problem
mu = 5 # mean breaking strength of the gunny bags of the population in kg/cm^2
sigma = 1.5 # standard deviation of the breaking strength of the gunny bags of the population in kg/cm^2

# Declaring the given parameters in the problem
min_val = mu - 4*sigma # assuming a min val of 4 std dev below the mean
max_val = mu + 4*sigma # assuming a max val of 4 std dev above the mean

# getting 10000 points for the x-axis from min-val to max-val
x = np.linspace(min_val,max_val,10000)
# getting the corresponding probility density value for x, considering a normal distribution with a mean of 5 and a standard deviation of 1.5
pdf = stats.norm.pdf(x, loc=mu, scale=sigma)



# #### 2.1 What proportion of the gunny bags have a breaking strength of less than 3.17 kg per sq cm?

# ##### **11.12%** of the gunny bags have a breaking strenght of less than 3.17 kg per sq. cm.

# In[9]:


# mu and sigma is defined above as 5 and 1.5 respectively
xbar = 3.17 # in kg/cm^2 as given in the question 2.1

auc_lequal_xbar = stats.norm.cdf(xbar,loc=mu,scale=sigma)
auc_lequal_xbar = round(auc_lequal_xbar,4)
print(f'The propotion of the gunny bags have a breaking strength of less than 3.17 kg per sq cm is: {auc_lequal_xbar*100}%')

#Lets plot the normal distribution of the breaking strength of the gunny bags
# mu and sigma is defined above
# the x and pdf values are defined above
x_fill = np.linspace(min_val,xbar,10000)
y_fill = stats.norm.pdf(x_fill, loc=mu, scale=sigma)

plt.figure(figsize=(10,6))
plt.title(f'Normal Distribution: mu:{mu} and sigma:{sigma} (in kg/cm^2)')
plt.ylabel('Probability Density')
plt.xlabel('Breaking Strength of Gunny Bags in kg/cm^2')
plt.plot(x,pdf)
plt.fill_between(x_fill,y_fill,color='r')
plt.axvline(x=xbar,linestyle='--',color='black',ymax=.48)
plt.legend(['pdf',f'cdf: {auc_lequal_xbar} P(x<={xbar})', f'xbar: x={xbar}'])
plt.savefig("./images/Breaking_Strength_less_Than_3_17.svg")
plt.show()


# #### 2.2 What proportion of the gunny bags have a breaking strength of at least 3.6 kg per sq cm.?

# ##### **82.47%** of the gunny bags have a breaking strenght of at least than 3.6 kg per sq. cm or higher.

# In[10]:


# mu and sigma is defined above as 5 and 1.5 respectively
xbar = 3.6 # in kg/cm^2 as given in the question 2.2

auc_gequal_xbar = 1 - stats.norm.cdf(xbar,loc=mu,scale=sigma)
auc_gequal_xbar = round(auc_gequal_xbar,4)
print(f'The proportion of the gunny bags have a breaking strength of at least 3.6 kg per sq cm is: {auc_gequal_xbar*100}%')

#Lets plot the normal distribution of the breaking strength of the gunny bags
# mu and sigma is defined above
# the x and pdf values are defined above
x_fill = np.linspace(xbar,max_val,10000)
y_fill = stats.norm.pdf(x_fill, loc=mu, scale=sigma)

plt.figure(figsize=(10,6))
plt.title(f'Normal Distribution: mu:{mu} and sigma:{sigma} (in kg/cm^2)')
plt.ylabel('Probability Density')
plt.xlabel('Breaking Strength of Gunny Bags in kg/cm^2')
plt.plot(x,pdf)
plt.fill_between(x_fill,y_fill,color='r')
plt.axvline(x=xbar,linestyle='--',color='black',ymax=.64)
plt.legend(['pdf',f'cdf: {auc_gequal_xbar} P(x>={xbar})', f'xbar: x={xbar}'])
plt.savefig("./images/Breaking_Strength_greater_Than_equal_to_3_6.svg")
plt.show()


# #### 2.3 What proportion of the gunny bags have a breaking strength between 5 and 5.5 kg per sq cm.?

# ##### **13.06%** of the gunny bags have a breaking strenght between 5 and 5.5 kg per sq. cm or higher.

# In[11]:


# mu and sigma is defined above as 5 and 1.5 respectively
xbar_low = 5 # in kg/cm^2 as given in the question 2.3
xbar_high = 5.5 # in kg/cm^2 as given in the question 2.3

auc_xbar_high = stats.norm.cdf(xbar_high,loc=mu,scale=sigma)
auc_xbar_low = stats.norm.cdf(xbar_low,loc=mu,scale=sigma)
auc_between_xhigh_xlow = auc_xbar_high-auc_xbar_low
auc_between_xhigh_xlow = round(auc_between_xhigh_xlow,4)
print(f'The proportion of the gunny bags have a breaking strength between 5 and 5.5 kg per sq cm is: {round(auc_between_xhigh_xlow*100,2)}%')

#Lets plot the normal distribution of the breaking strength of the gunny bags
# mu and sigma is defined above
# the x and pdf values are defined above
x_fill = np.linspace(xbar_low,xbar_high,10000)
y_fill = stats.norm.pdf(x_fill, loc=mu, scale=sigma)

plt.figure(figsize=(10,6))
plt.title(f'Normal Distribution: mu:{mu} and sigma:{sigma} (in kg/cm^2)')
plt.ylabel('Probability Density')
plt.xlabel('Breaking Strength of Gunny Bags in kg/cm^2')
plt.plot(x,pdf)
plt.fill_between(x_fill,y_fill,color='r')
plt.axvline(x=xbar_low,linestyle='--',color='black',ymax=.95)
plt.axvline(x=xbar_high,linestyle='--',color='black',ymax=.90)
plt.legend(['pdf',f'cdf: {auc_between_xhigh_xlow} P({xbar_low}<=x<={xbar_high})', f'xbar low: x={xbar_low}', f'xbar high: x={xbar_high}'])
plt.savefig("./images/Breaking_Strength_between_5and5_5.svg")
plt.show()


# #### 2.4 What proportion of the gunny bags have a breaking strength NOT between 3 and 7.5 kg per sq cm.?

# ##### **13.09%** of the gunny bags have a breaking strenght NOT between 3 and 7.5 kg per sq. cm or higher.

# In[12]:


# mu and sigma is defined above as 5 and 1.5 respectively
xbar_less = 3 # in kg/cm^2 as given in the question 2.3
xbar_more = 7.5 # in kg/cm^2 as given in the question 2.3

auc_xbar_less = stats.norm.cdf(xbar_less,loc=mu,scale=sigma)
auc_xbar_more = 1 - stats.norm.cdf(xbar_more,loc=mu,scale=sigma)
auc_xbar_less_or_more = auc_xbar_less + auc_xbar_more
auc_xbar_less_or_more = round(auc_xbar_less_or_more,4)
print(f'The proportion of the gunny bags have a breaking strength NOT between 3 and 7.5 kg per sq cm is: {round(auc_xbar_less_or_more*100,2)}%')

#Lets plot the normal distribution of the breaking strength of the gunny bags
# mu and sigma is defined above
# the x and pdf values are defined above
x_less_fill = np.linspace(min_val,xbar_less,10000)
y_less_fill = stats.norm.pdf(x_less_fill, loc=mu, scale=sigma)

x_more_fill = np.linspace(xbar_more,max_val,10000)
y_more_fill = stats.norm.pdf(x_more_fill, loc=mu, scale=sigma)

plt.figure(figsize=(10,6))
plt.title(f'Normal Distribution: mu:{mu} and sigma:{sigma} (in kg/cm^2)')
plt.ylabel('Probability Density')
plt.xlabel('Breaking Strength of Gunny Bags in kg/cm^2')
plt.plot(x,pdf)
plt.fill_between(x_less_fill,y_less_fill,color='r')
plt.fill_between(x_more_fill,y_more_fill,color='g')
plt.axvline(x=xbar_less,linestyle='--',color='black',ymax=.41)
plt.axvline(x=xbar_more,linestyle='--',color='black',ymax=.27)
plt.legend(['pdf',f'cdf: {round(auc_xbar_less,4)} P(x<={xbar_less})',f'cdf: {round(auc_xbar_more,4)} P(x>={xbar_more})', f'xbar lower: x={xbar_less}', f'xbar higher: x={xbar_more}'])
plt.savefig("./images/Breaking_Strength_not_between_3and7_5.svg")
plt.show()


# ## Problem 3
# ### Zingaro stone printing is a company that specializes in printing images or patterns on polished or unpolished stones. However, for the optimum level of printing of the image, the stone surface has to have a Brinell's hardness index of at least 150. Recently, Zingaro has received a batch of polished and unpolished stones from its clients. Use the data provided to answer the following (assuming a 5% significance level).

# In[13]:


# import the data provided for the batch recieved of polished and unpolished stones from its clients.
zingaro_stones_data = pd.read_csv('./data/Zingaro_Company.csv')


# In[14]:


# Explore data recieved.
zingaro_stones_data.head()


# #### 3.1 Zingaro has reason to believe that the unpolished stones may not be suitable for printing. Do you think Zingaro is justified in thinking so?

# ##### Given: **hypothesized mean in BHI mu = 150**, **level of significance alpha = 0.05**,**Number of observations = 75**,
# #####        **Mean of Unpolished stones = 134**, **Standard deviation of Unpolished stones = 33.04**
# ##### Since we have the sample mean and std and we are working with one sample will will calculate the t_statistic and run a t_test
# ##### Null Hypothesis for our t_test H_0 : Mean BHI of Unpolished stones >= 150 BHI, which would mean there are suitable for printing.
# ##### Alternate Hypothesis for our t_test H_A : Mean BHI of Unpolished stones < 150 BHI, which would mean there are NOT suitable for printing.
# ##### We call calate the t-statistic using the formular t = (xbar-mu)/(s/sqrt(n)) where xbar is the sample mean, mu is the hypothesized mean, s is the standard deviation of the sample and n is the numbers of observations.
# ##### The value of the t-statistic=**-4.1646**
# ##### We use the t.cdf function of stats to get **pvalue**, the degree of freedom used for the function is **n-1**
# ##### The pvalue=**0.0000417**, which is below the alpha value of 0.05 so we reject the Null Hypothesis "H_0 : Mean BHI of Unpolished stones >= 150 BHI"
# ##### We accept the Alternate Hypothesis that H_A : Mean BHI of Unpolished stones < 150 BHI, which would mean there are NOT suitable for printing.
# 
# 
# 
# 
# 
# 

# In[15]:


# check for null or zeros
zingaro_stones_data.info() # check for nulls, no nulls
zingaro_stones_data.isnull().sum() # check for nulls, no nulls
zingaro_stones_data.shape # check number of observations 75.


# In[16]:


zingaro_stones_data.describe()


# In[17]:


# Given Data Points.
# 1. for optimum printing the BHI(Brinell's hardness index) is equal or greater than 150
mu = 150 # hypothesized mean in BHI
# 2. for our tests the significance level or the Type I error tolerated is 5%, we are looking or a 95% confidance level test.
alpha = 0.05
# 3. number of observations in the sample batch for Unpolished stones given are 75.
n= 75

# State the Hypothesis:
# For unpolished stones:
# H_0 : Mean BHI of Unpolished stones >= 150 BHI
# H_A : Mean BHI of Unpolished stones < 150 BHI


# Assumptions, reference : https://www.financestrategists.com/wealth-management/fundamental-vs-technical-analysis/central-limit-theorem/
# 1. The samples are independent from each other, there no dependancy of one sample over the other
# 2. The samples are randomly selected form all the samples/batches recieved from the client.
# 3 The sample size is >= 30.
# we can apply the Centeral Limit Theorem, to suggest that the sampling mean BHI should lie in a distribution which follows a normal distribution curve.

# calculate the mean BHI for the unpolished stones sample
x_unpolished = zingaro_stones_data['Unpolished '].mean()

# calculate the standard deviation for the unpolished stones sample
s_unpolished = zingaro_stones_data['Unpolished '].std()
sigma = s_unpolished/np.sqrt(n)

# since we are woking with the sample mean and sample standard deviation we will used the t-test to do the hypothesis tests.
# we will run a one sample t-test.

# Calculate the t statistic
# reference https://www.calculatorway.com/t-statistic-calculator/
# for this we would need to calculate the t-satistic using the formular t = (xbar-mu)/(s/sqrt(n))
# where xbar = sample mean, mu = hypothesized mean, s= sample standard deviation and n = number of observations in the sample

t_unpolished = (x_unpolished - mu)/(s_unpolished/np.sqrt(n))
print(t_unpolished)
# Find the p_value corresponding the the t statistic for a one tailed test.
# reference: https://stackoverflow.com/questions/17559897/python-p-value-from-t-statistic
# import the libaries needed
from scipy.stats import t

# get the degrees of freedom
dof = len(zingaro_stones_data['Unpolished ']) - 1

# get the p_value
p_value = t.cdf(t_unpolished, dof)
print(p_value)

if p_value > alpha:
    print('H_0 : Mean BHI of Unpolished stones >= 150 BHI, Fail to Reject')
else:
    print('H_0 : Mean BHI of Unpolished stones >= 150 BHI, Is Rejected')
    print('H_1 : Mean BHI of Unpolished stones < 150 BHI, Is Possiblily True')



print('With a 95% confidance level or for a 5% signaficance level, we can say that the belief by Zingaro "that the unpolished stones may not be suitable for printing" in this instance is true.')
print(f'There is only a {round(p_value*100,4)}%, chance based on the given sample that the mean BHI of Unpolished stones is >= 150 BHI.')

#polting a t-distribution with dof of "74" and mean of  '134.11' and standard deviation of '33.04'
min_val = x_unpolished - 5 * s_unpolished
max_val = x_unpolished + 5 *  s_unpolished
x = np.linspace(min_val,max_val,1000)
y = t.pdf(x,dof,loc=mu, scale=sigma)
t_critical = t.ppf(0.05,dof,loc=mu,scale=sigma)
x_fill = np.linspace(min_val,t_critical,1000)
y_fill = t.pdf(x_fill,dof,loc=mu, scale=sigma)
x_fill_a = np.linspace(t_critical,max_val,1000)
y_fill_a = t.pdf(x_fill_a,dof,loc=mu, scale=sigma)
plt.figure(figsize=(10,6))
plt.plot(x,y)
plt.title(f'T Distribution: mu:{round(mu,2)} and sigma:{round(sigma,2)} and dof:{dof}')
plt.ylabel('Probability Density')
plt.xlabel("Hardness of Unpolished rock measured using Brinell's hardness index(BHI)")
plt.fill_between(x_fill,y_fill,color='r')
plt.fill_between(x_fill_a,y_fill_a,color='g')
plt.axvline(x=t_critical,linestyle='--',color='b',ymax=.40)
plt.axvline(x=t_unpolished,linestyle='--',color='y',ymax=.40)
plt.legend(['t_pdf','Rejection region','Fail to Reject region',f't_critical value(x={round(t_critical)})',f't satistic for test(x={round(t_unpolished)})'])
plt.savefig("./images/Unpolished_Stones_unsuitable.svg")
plt.show()


# #### 3.2 Is the mean hardness of the polished and unpolished stones the same?

# ##### We will run the stats.ttest_ind test to check if the two samples have the same mean.
# ##### Assumptions for this test are:
# ##### We will run all test with a alpha value of 0.05
# ##### 1. Independance of the observations: The Assumption is that the observations here are independant of each other.
# ##### 2. No Siginficant outliers in the two samples
# ##### 3. Normality of the samples
# ##### 4. Homogeneity of variances 
# ##### We test of these, there are ouliers since there are only four outliers we treat them but setting the outlier values to 1.5*IQR
# ##### We run the Shapiro-Wilk test to check for normalitiy the test shows that for both the samples, the data is consistent with a normal distribution.
# ##### We run the Lavenes test to check for Homogeneity of variances, the Lavenes test show that "The variance are not equal for at least one pair of samples", since we have only two samples, the variance is not equal between them.
# #### We now run the stats.ttest_ind test to check if the means between the two are the same, we use the parameter, equal_var=False as the variances between the samples were not the same.
# #### Results: the pvalue for the stats.ttest_ind was '00156' which is less than alpha of '0.05', so we reject the null hypotheses and accept the alternate of H_A : The means are significantly different, This tell us that the mean hardness of the polished and unpolished stones are significantly different 
# 
# 

# In[18]:


#Given Data points:
#1. sample of 75 observations of polished stones
polished = zingaro_stones_data['Treated and Polished']
#2. sample of 75 observations of unpolished stones
unpolished = zingaro_stones_data['Unpolished ']#Since we have two samples measuring the same quantity in this case it is the Hardness of the material using BHI(Brinell's hardness index).
# we can look at using the independant two sample t test to check if both the samples have the same mean or not.

#Assumptions for the test
# Reference : https://www.datanovia.com/en/lessons/t-test-assumptions/independent-t-test-assumptions/#:~:text=The%20two-samples%20independent%20t-test%20assume%20the%20following%20characteristics,group.%20No%20significant%20outliers%20in%20the%20two%20groups
#1. Independance of the observations: The Assumption is that the observations here are independant of each other.
#2. No Siginficant outliers in the two samples
#3. Normality of the samples
#4. Homogeneity of variances 

# Treating the Outliers.
# we have three outlier values in the dataset for polished and one for dataset for unpolished
data=[polished,unpolished]
plt.boxplot(data,labels=['Polished','Unpolished'])
plt.show()
# we treat the outliers by replacing the values with 1.5 IQR values.
def get_iqr_max(q1,q3,val):
    iqr = q3-q1
    mod = 1.5*iqr
    if val <= q1 - mod:
        val = q1 - mod
        return val
    if val >= q3 + mod:
        val = q3 + mod
        return val
    return val


q1 = np.percentile(polished,25)
q3 = np.percentile(polished,75)
zingaro_stones_data['Polished_Cleaned'] = zingaro_stones_data.apply(lambda row: get_iqr_max(q1,q3,row['Treated and Polished']),axis=1)
polished = zingaro_stones_data['Polished_Cleaned']
polished.name='Polished'
q1 = np.percentile(unpolished,25)
q3 = np.percentile(unpolished,75)
zingaro_stones_data['Unpolished_Cleaned'] = zingaro_stones_data.apply(lambda row: get_iqr_max(q1,q3,row['Unpolished ']),axis=1)
unpolished = zingaro_stones_data['Unpolished_Cleaned']
unpolished.name='Unpolished'

data = [polished,unpolished]
plt.boxplot(data,labels=['Polished','Unpolished'])
plt.show()


# In[19]:


# Test for Normality of the data for the samples
# We will do the Shapiro-Wilk test to check for normality of the samples for  polished_cleaned and unpolished_cleaned datasets
# we will use a significance level of 0.05.

# Hypothesis:
# H_0 : The data is consistant with a normal distribution.
# H_A : The data is NOT consistant with a normal distribution.
alpha = 0.05
from scipy import stats 

polished_cleaned_result = stats.shapiro(polished)
unpolished_cleaned_result = stats.shapiro(unpolished)

polished_p_val = polished_cleaned_result.pvalue
unpolished_p_val = unpolished_cleaned_result.pvalue

if polished_p_val > alpha:
    print('for polished, H_0 : The data is consistent with a normal distribution.')
else:
    print('for polished, H_A : The data is NOT consistent with a normal distribution.')
    
if unpolished_p_val > alpha:
    print('for unpolished, H_0 : The data is consistent with a normal distribution.')
else:
    print('for unpolished, H_A : The data is NOT consistent with a normal distribution.')


# In[20]:


# Test for equality in variances between the samples
# We will do the Levene test to check for equality in variances between the samples
# we will use a significance level of 0.05.

# Hypothesis:
# H_0 : The variance are equal across all samples.
# H_A : The variance are not equal for at least one pair of samples.
two_sample_result = stats.levene(polished,unpolished)
levene_result_p_val = two_sample_result.pvalue
if levene_result_p_val > alpha:
    print('H_0 : The variance are equal across all samples.')
else:
    print('H_A : The variance are not equal for at least one pair of samples.')


# In[21]:


# Now that we have tested all the Assumptions, can apply the ttest_ind
#1. Independance of the observations: The Assumption is that the observations here are independant of each other.
#2. No Siginficant outliers in the two samples: Outliers were treated to take on the iqr*1.5 of q1 and q3 where needed.
#3. Normality of the samples: Checked post outlier treatment using the "Shapiro-Wilk" test, found to be derived from a normal distribution with a 95% confidance level.
#4. Homogeneity of variances: Checked post outlier treatment using the "Levene" test, found the variance are not equal across the samples.

#Using the Two Sample Independant T test to check if "The mean hardness of the polished and unpolished stones the same"

# Hypothesis:
# H_0 : There No significant difference in means, or means are equal.
# H_A : The means are significantly different.
# Assuming a level of significance of 0.05
alpha = 0.05
result = stats.ttest_ind(polished,unpolished, equal_var=False) # equal_var=False is taken as the variance are not equal across the samples, Welch's t-test is done in this senario
# reference https://en.wikipedia.org/wiki/Welch%27s_t-test

print(result)
if result.pvalue > alpha:
    print('H_0 : There No significant difference in means, or means are equal.')
else:
    print('H_A : The means are significantly different.')


# In[22]:


# Visualing the analysis
# hisplots of the datasets

plt.figure(figsize=(10,6))
sns.histplot([polished,unpolished],kde=True,palette='viridis')
plt.axvline(x=polished.mean(),linestyle='--',c='#0000ff',ymax=.75)
plt.axvline(x=unpolished.mean(),linestyle='--',c='#00ff00',ymax=.35)
plt.legend(['pdf Unpolished','pdf Polished',f'Mean Polished: {round(polished.mean(),2)}',f'Mean Unpolished: {round(unpolished.mean(),2)}','Polished','Unpolished'])
plt.title("Histogram of Polished and Unpolished Stones Brinell's hardness index")
plt.xlabel("Hardness of Unpolished rock measured using Brinell's hardness index(BHI)")
plt.savefig("./images/polished_unpolished_hist.svg")
plt.show()




# In[23]:


dof = result.df
t_critical = result.statistic
x = np.linspace(-5,5,1000)
y = t.pdf(x,dof)
t_critical_left = t.ppf(0.025,dof)
rejection_left_x = np.linspace(-5,t_critical_left,1000)
rejection_left_y = t.pdf(rejection_left_x,dof)
t_critical_right = t.ppf(1-0.025,dof)
rejection_right_x = np.linspace(t_critical_right,5,1000)
rejection_right_y = t.pdf(rejection_right_x,dof)

plt.figure(figsize=(10,6))
plt.plot(x,y)
plt.fill_between(rejection_left_x,rejection_left_y,color='r')
plt.axvline(x=t_critical_left,linestyle='--',c='r')
plt.fill_between(rejection_right_x,rejection_right_y,color='r')
plt.axvline(x=t_critical_right,linestyle='--',c='r')
plt.axvline(x=t_critical,linestyle='--',c='m')
plt.legend(['pdf','Rejection Region','t-critical','_','_','t-sat'])
plt.ylabel('Probility Density')
plt.xlabel('t-values')
plt.title(f'T-Distribution: dof:{round(dof)}')
plt.savefig("./images/ttest_ind_distribution.svg")
plt.show()


# ## Problem 4
# ### Dental implant data: The hardness of metal implants in dental cavities depends on multiple factors, such as the method of implant, the temperature at which the metal is treated, the alloy used as well as the dentists who may favor one method above another and may work better in his/her favorite method. The response is the variable of interest.

# In[24]:


df = pd.read_excel('./data/Dental_Hardness_data.xlsx')


# In[25]:


df.head()


# In[26]:


df.info() # check for nulls, no nulls
df.shape # check number of observations 75.
df.isnull().sum() # check for nulls, no nulls


# In[27]:


df.Dentist.value_counts()


# In[28]:


df.Method.value_counts()


# In[29]:


df.Alloy.value_counts()


# In[30]:


df.Temp.value_counts()


# In[31]:


#Change all the int64 which are categorical variables to categories
#reference: https://bobbyhadz.com/blog/pandas-change-column-type-to-categorical#:~:text=To%20change%20the%20column%20type%20to%20Categorical%20in,selected%20column%2C%20passing%20it%20%22category%22%20as%20a%20parameter.

cols = ['Dentist','Method','Alloy','Temp']
for col in cols:
    df[col] = df[col].astype('category')



# In[32]:


df.info()


# In[33]:


# for the questions we are about to explore, we would be running ANOVA Tests both One Way and Two Way
# There are few Assumptions for these tests
#Assumptions for the test
# Reference : https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Mostly_Harmless_Statistics_%28Webb%29/11%3A_Analysis_of_Variance/11.03%3A_Two-Way_ANOVA_%28Factorial_Design%29
# The populations are normal.
# The observations are independent.
# The variances from each population are equal.
# The groups must have equal sample sizes.

# let us run these test and get to know if these assumptions are good.
# we will use the Shapiro-Wilk test to check for normality
def get_shap_result(alpha,p_val,col,level=None):
    if level == None:
        if p_val > alpha:
            print(f'for {col}, H_0 : The data is consistent with a normal distribution., with alpha:{alpha} and pvalue:{p_val}')
            return
        else:
            print(f'for {col}, H_A : The data is NOT consistent with a normal distribution., with alpha:{alpha} and pvalue:{p_val}')
            return        
    else:
        if p_val > alpha:
            print(f'for {col} and level {level}, H_0 : The data is consistent with a normal distribution., with alpha:{alpha} and pvalue:{p_val}')
            return
        else:
            print(f'for {col} and level {level}, H_A : The data is NOT consistent with a normal distribution., with alpha:{alpha} and pvalue:{p_val}')
            return
    

def check_norm(alpha,df,col,var,isfactor=False):
    if not isfactor:
        result = stats.shapiro(df[var])
        p_val = result.pvalue
        get_shap_result(alpha,p_val,col)
        return
    else:
        levels = df[col].value_counts().index
        temp_df = df[[col,var]].reset_index()
        for level in levels:
            temp_df_level = temp_df[temp_df[col]==level]
            result = stats.shapiro(temp_df_level[var])
            p_val = result.pvalue
            get_shap_result(alpha,p_val,col,level=level)
        return
check_norm(0.05,df,'Response','Response')
check_norm(0.05,df,'Dentist','Response',True)
check_norm(0.05,df,'Method','Response',True)
check_norm(0.05,df,'Alloy','Response',True)
check_norm(0.05,df,'Temp','Response',True)


# In[34]:


# As we see from the above the data does not follow the Normality assumption for all levels
# so we will see if transforming the target/dependant variable "Response" helps meet this assumption.
# Square root transform.
df['Response_root'] = df['Response'].apply(np.sqrt)
check_norm(0.05,df,'Response_root','Response_root')
check_norm(0.05,df,'Dentist','Response_root',True)
check_norm(0.05,df,'Method','Response_root',True)
check_norm(0.05,df,'Alloy','Response_root',True)
check_norm(0.05,df,'Temp','Response_root',True)


# In[35]:


# Cube root transform.
df['Response_3root'] = df['Response'].apply(np.cbrt)
check_norm(0.05,df,'Response_root','Response_3root')
check_norm(0.05,df,'Dentist','Response_3root',True)
check_norm(0.05,df,'Method','Response_3root',True)
check_norm(0.05,df,'Alloy','Response_3root',True)
check_norm(0.05,df,'Temp','Response_3root',True)


# In[36]:


# log transform.
df['Response_log'] = df['Response'].apply(np.log)
check_norm(0.05,df,'Response_log','Response_log')
check_norm(0.05,df,'Dentist','Response_log',True)
check_norm(0.05,df,'Method','Response_log',True)
check_norm(0.05,df,'Alloy','Response_log',True)
check_norm(0.05,df,'Temp','Response_log',True)


# In[37]:


# box cox transform sklearn
#https://sixsigmastudyguide.com/box-cox-transformation/
from sklearn.preprocessing import PowerTransformer as pt
import statsmodels.api as sm
df['Response_boxcox_pt'] = pt('box-cox').fit_transform(df['Response'].values.reshape(-1,1))
check_norm(0.05,df,'Response_boxcox_pt','Response_boxcox_pt')
check_norm(0.05,df,'Dentist','Response_boxcox_pt',True)
check_norm(0.05,df,'Method','Response_boxcox_pt',True)
check_norm(0.05,df,'Alloy','Response_boxcox_pt',True)
check_norm(0.05,df,'Temp','Response_boxcox_pt',True)


# In[38]:


#Check the Skew and Kurtosis for all the Transformations done
print(f'Response, Skew:',df['Response'].skew(), 'Kurt:',df['Response'].kurt())
print(f'Response_root, Skew:',df['Response_root'].skew(), 'Kurt:',df['Response_root'].kurt())
print(f'Response_3root, Skew:',df['Response_3root'].skew(), 'Kurt:',df['Response_3root'].kurt())
print(f'Response_log, Skew:',df['Response_log'].skew(), 'Kurt:',df['Response_log'].kurt())
print(f'Response_boxcox_pt, Skew:',df['Response_boxcox_pt'].skew(), 'Kurt:',df['Response_boxcox_pt'].kurt())
#We see that the Response_boxcox_pt gives the least skew 
#Also the Kurtosis for the same tell us that the distribution is flat and has thin tails
#After Transformation too the data is Not consistently Normal for all Factors and levels, but since the "Response_boxcox_pt" in the closest to normality Skew: 0.098, we will use this for the Questions following


# In[39]:


# Checking for the last assumption 
# The variances from each population are equal.
# Test for equality in variances between the samples
# We will do the Levene test to check for equality in variances between the samples
# we will use a significance level of 0.05.

# Hypothesis:
# H_0 : The variance are equal across all samples.
# H_A : The variance are not equal for at least one pair of samples.
alpha=0.05

def level_test_result(alpha,p_value):
    if p_value > alpha:
        print('H_0 : The variance are equal across all samples.')
    else:
        print('H_A : The variance are NOT equal for at least one pair of samples.')
    


Dentist_1 = df[df['Dentist']==1]['Response_boxcox_pt']
Dentist_2 = df[df['Dentist']==2]['Response_boxcox_pt']
Dentist_3 = df[df['Dentist']==3]['Response_boxcox_pt']
Dentist_4 = df[df['Dentist']==4]['Response_boxcox_pt']
Dentist_5 = df[df['Dentist']==5]['Response_boxcox_pt']

lev_result =  stats.levene(Dentist_1,Dentist_2,Dentist_3,Dentist_4,Dentist_5)
lev_p_value = lev_result.pvalue
print('Dentist')
level_test_result(alpha,lev_p_value)
print(lev_p_value)

Method_1 = df[df['Method']==1]['Response_boxcox_pt']
Method_2 = df[df['Method']==2]['Response_boxcox_pt']
Method_3 = df[df['Method']==3]['Response_boxcox_pt']
lev_result =  stats.levene(Method_1,Method_2,Method_3)
lev_p_value = lev_result.pvalue
print('Method')
level_test_result(alpha,lev_p_value)
print(lev_p_value)

Alloy_1 = df[df['Alloy']==1]['Response_boxcox_pt']
Alloy_2 = df[df['Alloy']==2]['Response_boxcox_pt']
lev_result =  stats.levene(Alloy_1,Alloy_2)
lev_p_value = lev_result.pvalue
print('Alloy')
level_test_result(alpha,lev_p_value)
print(lev_p_value)

Temp_1500 = df[df['Temp']==1500]['Response_boxcox_pt']
Temp_1600 = df[df['Temp']==1600]['Response_boxcox_pt']
Temp_1700 = df[df['Temp']==1700]['Response_boxcox_pt']
lev_result =  stats.levene(Temp_1500,Temp_1600,Temp_1700)
lev_p_value = lev_result.pvalue
print('Temp')
level_test_result(alpha,lev_p_value)
print(lev_p_value)

print('All')
lev_result =  stats.levene(Dentist_1,Dentist_2,Dentist_3,Dentist_4,Dentist_5,Method_1,Method_2,Method_3,Alloy_1,Alloy_2,Temp_1500,Temp_1600,Temp_1700)
lev_p_value = lev_result.pvalue
level_test_result(alpha,lev_p_value)
print(lev_p_value)

print('Dentist-Method')
lev_result =  stats.levene(Dentist_1,Dentist_2,Dentist_3,Dentist_4,Dentist_5,Method_1,Method_2,Method_3)
lev_p_value = lev_result.pvalue
level_test_result(alpha,lev_p_value)
print(lev_p_value)

print('Dentist-Method-Alloy')
lev_result =  stats.levene(Dentist_1,Dentist_2,Dentist_3,Dentist_4,Dentist_5,Method_1,Method_2,Method_3,Alloy_1,Alloy_2)
lev_p_value = lev_result.pvalue
level_test_result(alpha,lev_p_value)
print(lev_p_value)


# #### 4.1 How does the hardness of implants vary depending on dentists?

# ##### The hardness of the implants are the same and do not vary significantly depending on dentist

# In[45]:


#Let use plot the hardness of implants (response) by dentists with point plot
#We will use the "Response_boxcox_pt" as it is the once with the least skew
plt.figure(figsize=(10,6))
sns.pointplot(df[['Dentist','Response']],y='Response',x='Dentist',errorbar=('ci', False))
plt.savefig("./images/Dentist_point_plot.svg")
plt.show()

#Let use check the means hardness by dentist
means_dentist = df.groupby('Dentist')[['Response','Response_boxcox_pt']].mean()
std_dentist = df.groupby('Dentist')[['Response','Response_boxcox_pt']].std()
means_dentist.rename(columns={'Response':'Mean','Response_boxcox_pt':'Mean_Box_Cox'},inplace=True)
std_dentist.rename(columns={'Response':'STD','Response_boxcox_pt':'STD_Box_Cox'},inplace=True)
dentist = pd.concat([means_dentist,std_dentist],axis=1)
print(dentist)
#The mean hardness for Dentist 1,2 seem to be close to each other and the highest
#The mean hardness for Dentist 3,4 seem to be in the mid range
#Then mean hardness for Dentist 5 is the least

#Let us check if the mean hardness for the Dentist are different from each other significantly
#We will run a One Way ANOVA Test on this but will use the "Response_boxcox_pt" as this is least skewed.
#Also based on the Levenes test for the 'Dentist' we know that the varence is Not Equal.
#We go ahead with the ANOVA for now but mak a note that we may need larger sample sizes in future to mitigate the variance not being equal and the fact that the data is sligthly skewed
#we will consider a alha of 0.05
# Hypothesis:
# H_0 : All the samples have the same population mean.
# H_A : All the samples DO NOT have the same population mean.
alpha = 0.05
results= stats.f_oneway(Dentist_1,Dentist_2,Dentist_3,Dentist_4,Dentist_5)
print(results)
pvalue = results.pvalue

if pvalue>alpha:
    print('We can NOT reject H0: All the samples have the same population mean.')
else:
    print('We can reject H0, HA: All the samples DO NOT have the same population mean.')

print('The hardness of the implants are the same and do not vary significantly depending on dentist')


# #### 4.2 How does the hardness of implants vary depending on methods?

# ##### The hardness of implants do not vary between Methods 1 and 2, it varies significantly for Method 3 and on an average is lower compared to Methods 1 and 2.

# In[47]:


#Let use plot the hardness of implants (response) by method with point plot
#We will use the "Response_boxcox_pt" as it is the once with the least skew
plt.figure(figsize=(10,6))
sns.pointplot(df[['Method','Response']],y='Response',x='Method',errorbar=('ci', False))
plt.savefig("./images/Method_point_plot.svg")
plt.show()

#Let use check the means hardness by dentist
means_method = df.groupby('Method')[['Response','Response_boxcox_pt']].mean()
std_method = df.groupby('Method')[['Response','Response_boxcox_pt']].std()
means_method.rename(columns={'Response':'Mean','Response_boxcox_pt':'Mean_Box_Cox'},inplace=True)
std_method.rename(columns={'Response':'STD','Response_boxcox_pt':'STD_Box_Cox'},inplace=True)
method = pd.concat([means_method,std_method],axis=1)
print(method)
#The mean hardness for Method 1,2 seem to be close to each other and the highest
#The mean hardness for Method 3 is the least

#Let us check if the mean hardness for the Method are different from each other significantly
#We will run a One Way ANOVA Test on this but will use the "Response_boxcox_pt" as this is least skewed.
#Also based on the Levenes test for the 'Method' we know that the varence is Equal.
#We go ahead with the ANOVA for now but mak a note that we may need larger sample sizes in future to mitigate the fact that the data is sligthly skewed
#we will consider a alha of 0.05
# Hypothesis:
# H_0 : All the samples have the same population mean.
# H_A : All the samples DO NOT have the same population mean.
alpha = 0.05
results= stats.f_oneway(Method_1,Method_2,Method_3)
print(results)
pvalue = results.pvalue

if pvalue>alpha:
    print('We can NOT reject H0: All the samples have the same population mean.')
else:
    print('We can reject H0, HA: All the samples DO NOT have the same population mean.')
#Since we know from above that All the samples DO NOT have the same population mean, we will run the Tukey’s HSD test, which will tell us for which combinations of the Methods does the means vary
from statsmodels.stats.multicomp import pairwise_tukeyhsd

result = pairwise_tukeyhsd(df['Response_boxcox_pt'],df['Method'])
print(result)
#This Test tell us that the means for Method 1 and Method 2 are NOT significantely different.
#This Test tell us that the means for Method 1 and Method 3 are significantely different.
#This Test tell us that the means for Method 2 and Method 3 are significantely different.
print('The hardness of implants do not vary between Methods 1 and 2, it varies significantly for Method 3 and on an average is lower compared to Methods 1 and 2.')


# #### 4.3 What is the interaction effect between the dentist and method on the hardness of dental implants for each type of alloy?

# ##### The interaction between dentist and the method used for Alloy 1 significantly impacts the reponse or hardness of the implants.
# ##### The interaction between dentist and the method used for Alloy 2 significantly impacts the reponse or hardness of the implants,But to a lesser Degree than it is impacted for Alloy 1.

# In[55]:


#Let use plot the hardness of implants (response) by Dentist and method with point plot
#We will use the "Response_boxcox_pt" as it is the once with the least skew


# Creating sperate dataset per Alloy Type:
df_Alloy1 = df[df['Alloy']==1]
df_Alloy2 = df[df['Alloy']==2]

plt.figure(figsize=(10,6))
sns.pointplot(df_Alloy1[['Dentist','Method','Response']],y='Response',x='Dentist',hue='Method',errorbar=('ci', False))
plt.savefig("./images/Alloy1.svg")
plt.show()
plt.figure(figsize=(10,6))
sns.pointplot(df_Alloy2[['Dentist','Method','Response']],y='Response',x='Dentist',hue='Method',errorbar=('ci', False))
plt.savefig("./images/Alloy2.svg")
plt.show()

#It seems that there could be some interaction between Method and Dentist, as the in cases of Method 1 and two there is intersection of the dataset
#Let us run a two way ANOVA on the interaction between Dentist and Method
from statsmodels.formula.api import ols

model = ols('Response_boxcox_pt~C(Dentist):C(Method)',data=df_Alloy1).fit()
print(sm.stats.anova_lm(model,typ=2))
#Looking at the result of the ANOVA Test
print('The interaction between dentist and the method used for Alloy 1 significantly impacts the reponse or hardness of the implants.')

model = ols('Response_boxcox_pt~C(Dentist):C(Method)',data=df_Alloy2).fit()
print(sm.stats.anova_lm(model,typ=2))
#Looking at the result of the ANOVA Test
print('The interaction between dentist and the method used for Alloy 2 significantly impacts the reponse or hardness of the implants,But to a lesser Degree than it is impacted for Alloy 1')


# #### 4.4 How does the hardness of implants vary depending on dentists and methods together?

# #### Dentist 1 and 4, gets the best reponse for hardness Using Method 2
# #### Dentist 2 and 5, gets the best reponse for hardness Using Method 1
# #### Dentist 3, gets the best reponse for hardness Using Method 3

# In[65]:


plt.figure(figsize=(10,6))
sns.pointplot(df[['Dentist','Method','Response']],y='Response',x='Dentist',hue='Method',errorbar=('ci', False))
plt.savefig("./images/Alloy2.svg")
plt.show()

model = ols('Response_boxcox_pt~C(Dentist)+C(Method)',data=df).fit()
print(sm.stats.anova_lm(model,typ=2))
print('Looking at each dentist and each methods together WITH OUT interactions we see that, dentists do not significatly effect the hardness of the implants but the different methods do.')


model = ols('Response_boxcox_pt~C(Dentist):C(Method)',data=df).fit()
print(sm.stats.anova_lm(model,typ=2))
print('Looking at dentists and methods together with interactions we see that, dentist using a specific method have a signification impact on the Hardness.')

model = ols('Response_boxcox_pt~C(Dentist)+C(Method)+C(Dentist):C(Method)',data=df).fit()
print(sm.stats.anova_lm(model,typ=2))
print('Looking at each dentist and each methods together WITH OUT interactions we see that, dentists do not significatly effect the hardness of the implants but the different methods do.')


model = ols('Response_boxcox_pt~C(Method)+C(Dentist):C(Method)',data=df).fit()
print(sm.stats.anova_lm(model,typ=2))
print('Looking at each dentist and each methods together WITH OUT interactions we see that, dentists do not significatly effect the hardness of the implants but the different methods do.')


print('Dentist 1 and 4, gets the best reponse for hardness Using Method 2')
print('Dentist 2 and 5, gets the best reponse for hardness Using Method 1')
print('Dentist 3, gets the best reponse for hardness Using Method 3')



# In[ ]:




