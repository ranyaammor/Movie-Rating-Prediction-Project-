#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats.mstats import winsorize
import seaborn as sns


# In[2]:


#load data
df = pd.read_csv("C:/Users/ranya/OneDrive/Bureau/DATA ANALYTICS/DATA MINING/Project.csv")
df.info()


# In[3]:


# Convert 'datadate' column to datetime format
df['date_in_question'] = pd.to_datetime(df['date_in_question'])
# Extract year from 'datadate' and create a new 'year' column
df['year'] = df['date_in_question'].dt.year
# Display the updated DataFrame with 'year' columnn
df


# In[4]:


#drop rows with missing values
df.dropna(subset = ['saleq_lag1','atq_lag1','piq_lag1','teqq_lag1','niq_lag1','saleq_lag2','atq_lag2','piq_lag2','teqq_lag2','niq_lag2','saleq_lag3','atq_lag3','piq_lag3','teqq_lag3','niq_lag3'],inplace = True)
# Ensure that variables intended to be used as denominators in ratio calculations not equal to zero
df = df[(df['saleq_lag1'] != 0) & (df['atq_lag1'] != 0) & (df['piq_lag1'] != 0)  & (df['teqq_lag1'] != 0)& (df['niq_lag1'] != 0) & (df['saleq_lag2'] != 0) & (df['atq_lag2'] != 0) & (df['piq_lag2'] != 0)  & (df['teqq_lag2'] != 0)& (df['niq_lag2'] != 0) & (df['saleq_lag3'] != 0) & (df['atq_lag3'] != 0) & (df['piq_lag3'] != 0)  & (df['teqq_lag3'] != 0)& (df['niq_lag3'] != 0)]
df.shape


# In[5]:


#calculate ratios
#profit margin lag 1
df['PM_lag1'] = df['niq_lag1']/df['saleq_lag1']
#return on asset
df['ROA_lag1'] = df['niq_lag1']/df['atq_lag1']
#asset turnover
df['TAT_lag1'] = df['saleq_lag1']/df['atq_lag1']
#tax burden
df['TB_lag1'] = df['niq_lag1']/df['piq_lag1']
#equity numtiplier
df['EM_lag1'] = df['atq_lag1'] / df['teqq_lag1']
#market to book
df['ROE_lag1'] = df['niq_lag1'] / df['teqq_lag1']


# In[6]:


#calculate ratios
#profit margin lag 2
df['PM_lag2'] = df['niq_lag2']/df['saleq_lag2']
#return on asset
df['ROA_lag2'] = df['niq_lag2']/df['atq_lag2']
#asset turnover
df['TAT_lag2'] = df['saleq_lag2']/df['atq_lag2']
#tax burden
df['TB_lag2'] = df['niq_lag2']/df['piq_lag2']
#equity numtiplier
df['EM_lag2'] = df['atq_lag2'] / df['teqq_lag2']
#market to book
df['ROE_lag2'] = df['niq_lag2'] / df['teqq_lag2']


# In[7]:


#calculate ratios
#profit margin lag 3
df['PM_lag3'] = df['niq_lag3']/df['saleq_lag3']
#return on asset
df['ROA_lag3'] = df['niq_lag3']/df['atq_lag3']
#asset turnover
df['TAT_lag3'] = df['saleq_lag3']/df['atq_lag3']
#tax burden
df['TB_lag3'] = df['niq_lag3']/df['piq_lag3']
#equity numtiplier
df['EM_lag3'] = df['atq_lag3'] / df['teqq_lag3']
#market to book
df['ROE_lag3'] = df['niq_lag3'] / df['teqq_lag3']


# In[8]:


perc = [.05,.95]
for v in ['PM_lag1','ROA_lag1','TAT_lag1','TB_lag1','EM_lag1','ROE_lag1']:
    df[v] = winsorize(df[v],limits=(0.05,0.05),inplace=True)
    
df[['PM_lag1','ROA_lag1','TAT_lag1','TB_lag1','EM_lag1','ROE_lag1']].describe(percentiles=perc).round(2)


# In[9]:


perc = [.05,.95]
for v in ['PM_lag2','ROA_lag2','TAT_lag2','TB_lag2','EM_lag2','ROE_lag2']:
    df[v] = winsorize(df[v],limits=(0.05,0.05),inplace=True)
    
df[['PM_lag2','ROA_lag2','TAT_lag2','TB_lag2','EM_lag2','ROE_lag2']].describe(percentiles=perc).round(2)


# In[10]:


perc = [.05,.95]
for v in ['PM_lag3','ROA_lag3','TAT_lag3','TB_lag3','EM_lag3','ROE_lag3']:
    df[v] = winsorize(df[v],limits=(0.05,0.05),inplace=True)
    
df[['PM_lag3','ROA_lag3','TAT_lag3','TB_lag3','EM_lag3','ROE_lag3']].describe(percentiles=perc).round(2)


# In[13]:


# Filter for bankrupt companies
df_bankrupt = df[df['bankrupt'] == 1]
# First, calculate the overall average for PM_lag1, PM_lag2, PM_lag3 for bankrupt companies.
avg_PM_lags = df_bankrupt[['PM_lag1', 'PM_lag2', 'PM_lag3']].mean()

plt.figure(figsize=(8, 6))
plt.plot(['PM_lag1', 'PM_lag2', 'PM_lag3'], avg_PM_lags, marker='o', linestyle='-', color='purple')
plt.ylabel('Average Profit Margin')
plt.title('Average Profit Margin by PM Lag')
plt.grid(axis='y')
plt.show()


# In[14]:


# Filter for bankrupt companies
df_bankrupt = df[df['bankrupt'] == 1]

# Calculate the overall average for ROE_lag1, ROE_lag2, ROE_lag3 for bankrupt companies.
avg_ROE_lags = df_bankrupt[['ROE_lag1', 'ROE_lag2', 'ROE_lag3']].mean()

# Plotting the average Return on Equity for ROE_lag1, ROE_lag2, ROE_lag3
plt.figure(figsize=(8, 6))
plt.plot(['ROE_lag1', 'ROE_lag2', 'ROE_lag3'], avg_ROE_lags, marker='o', linestyle='-', color='blue')
plt.ylabel('Average Return on Equity')
plt.title('Average Return on Equity by Lag for Bankrupt Companies')
plt.grid(axis='y')
plt.show()


# In[16]:


# Filter for bankrupt companies
df_bankrupt = df[df['bankrupt'] == 1]

# Calculate the overall average for ROE_lag1, ROE_lag2, ROE_lag3 for bankrupt companies.
avg_ROE_lags = df_bankrupt[['ROA_lag1', 'ROA_lag2', 'ROA_lag3']].mean()

# Plotting the average Return on Equity for ROE_lag1, ROE_lag2, ROE_lag3
plt.figure(figsize=(8, 6))
plt.plot(['ROA_lag1', 'ROA_lag2', 'ROA_lag3'], avg_ROE_lags, marker='o', linestyle='-', color='red')
plt.ylabel('Average Return on Asset')
plt.title('Average Return on Asset by Lag for Bankrupt Companies')
plt.grid(axis='y')
plt.show()


# In[18]:


# Filter for bankrupt companies
df_bankrupt = df[df['bankrupt'] == 1]

# Calculate the overall average for ROE_lag1, ROE_lag2, ROE_lag3 for bankrupt companies.
avg_ROE_lags = df_bankrupt[['EM_lag1', 'EM_lag2', 'EM_lag3']].mean()

# Plotting the average Return on Equity for ROE_lag1, ROE_lag2, ROE_lag3
plt.figure(figsize=(8, 6))
plt.plot(['EM_lag1', 'EM_lag2', 'EM_lag3'], avg_ROE_lags, marker='o', linestyle='-', color='orange')
plt.ylabel('Average Equity Multiplier')
plt.title('Average Equity Multiplier by Lag for Bankrupt Companies')
plt.grid(axis='y')
plt.show()


# In[20]:


# Filter for bankrupt companies
df_bankrupt = df[df['bankrupt'] == 1]

# Calculate the overall average for ROE_lag1, ROE_lag2, ROE_lag3 for bankrupt companies.
avg_ROE_lags = df_bankrupt[['TAT_lag1', 'TAT_lag2', 'TAT_lag3']].mean()

# Plotting the average Return on Equity for ROE_lag1, ROE_lag2, ROE_lag3
plt.figure(figsize=(8, 6))
plt.plot(['TAT_lag1', 'TAT_lag2', 'TAT_lag3'], avg_ROE_lags, marker='o', linestyle='-', color='black')
plt.ylabel('Average Total Asset Turnover')
plt.title('Average Total Asset Turnover by Lag for Bankrupt Companies')
plt.grid(axis='y')
plt.show()


# In[25]:


#Filter for bankrupt companies
df_bankrupt = df[df['bankrupt'] == 1]

# Calculate the overall average for ROE_lag1, ROE_lag2, ROE_lag3 for bankrupt companies.
avg_ROE_lags = df_bankrupt[['TB_lag1', 'TB_lag2', 'TB_lag3']].mean()

# Plotting the average Return on Equity for ROE_lag1, ROE_lag2, ROE_lag3
plt.figure(figsize=(8, 6))
plt.plot(['TB_lag1', 'TB_lag2', 'TB_lag3'], avg_ROE_lags, marker='o', linestyle='-', color='green')
plt.ylabel('Average Tax Burden')
plt.title('Average Tax Buden by Lag for Bankrupt Companies')
plt.grid(axis='y')
plt.show()


# In[28]:


# Filter for bankrupt companies
df_not_bankrupt = df[df['bankrupt'] == 0]
# First, calculate the overall average for PM_lag1, PM_lag2, PM_lag3 for bankrupt companies.
avg_PM_lags = df_not_bankrupt[['PM_lag1', 'PM_lag2', 'PM_lag3']].mean()

plt.figure(figsize=(8, 6))
plt.plot(['PM_lag1', 'PM_lag2', 'PM_lag3'], avg_PM_lags, marker='o', linestyle='-', color='purple')
plt.ylabel('Average Profit Margin')
plt.title('Average Profit Margin by PM Lag')
plt.grid(axis='y')
plt.show()

