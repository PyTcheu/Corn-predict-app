#!/usr/bin/env python
# coding: utf-8

# # Forecasting Milho B3

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
import os
from functools import reduce

from pandas import DataFrame
from io import StringIO
import time, json
import itertools
from datetime import date
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

import warnings
warnings.filterwarnings('ignore')


# ## Coletando dados

# In[2]:


weekly_files = []
monthly_files = []
daily_files = []
for filename in os.listdir("Data/"):
    if filename.endswith(".csv") :
        if filename.split('-')[-1:][0].split('.')[0] == ' W': 
            weekly_files.append(filename)
        elif filename.split('-')[-1:][0].split('.')[0] == ' M':
            monthly_files.append(filename)
        elif filename.split('-')[-1:][0].split('.')[0] == ' D':
            daily_files.append(filename)


# In[3]:


df_daily = []

for i in daily_files:
    df = pd.read_csv('Data/' + i)[['Data','Último']]
    df_daily.append(df)


# In[4]:


n = 0
for i in df_daily:
    #i['Data'] = i['Data'].astype('str').str.replace('.','-')
    #i['Último'] = i['Último'].astype('str').str.replace('.','').str.replace(',','.').astype('float')
    i.rename(columns={'Último':daily_files[n].split('.')[0]}, inplace=True)
    n += 1


# In[5]:


df_daily = reduce(lambda  left,right: pd.merge(left,right,on=['Data'],
                                            how='inner'), df_daily)

df_daily['Data'] = pd.to_datetime(df_daily['Data'], format='%d-%m-%Y')
df_daily = df_daily.set_index('Data')


# ## ARIMA com dados diários

# In[6]:


df_daily = df_daily.asfreq(freq='W', method='bfill')


# ### Teste adfuller

# In[7]:


def adfullter_transform(df):
    adfuller(df)
    
    result = adfuller(df) 

    print('ADF Statistic: %f' % result[0]) 
    print('p-value: %f' % result[1]) 
    
    df_diff = df - df.shift(periods=1)
    df_diff.dropna(inplace=True)

    result = adfuller(df_diff) 

    pval              = result[1]
    print('ADF Statistic: %f' % result[0]) 
    print('p-value: %f' % result[1]) 

    if pval < 0.05:
        print('Data is stationary')
        return df_diff
    
    else:
        print('Data after differencing is not stationary; so try log diff')
        
    


# In[8]:


result = adfuller(df_daily) 

print('ADF Statistic: %f' % result[0]) 
print('p-value: %f' % result[1]) 


# In[9]:


df_daily_diff = df_daily - df_daily.shift(periods=1)
df_daily_diff.dropna(inplace=True)

result = adfuller(df_daily_diff) 

pval              = result[1]
print('ADF Statistic: %f' % result[0]) 
print('p-value: %f' % result[1]) 

if pval < 0.05:
    print('Data is stationary')
else:
    print('Data after differencing is not stationary; so try log diff')


# ## Funções para benchmark de modelos

# In[10]:


def generate_date_series(start_date):
    end_date = [*start_date]
    end_date[0] = end_date[0] + 1
    
    dt = datetime.datetime(*start_date)
    end = datetime.datetime(*end_date)
    step = datetime.timedelta(days=7)

    result = []

    while dt < end:
        result.append(dt.strftime('%Y-%m-%d'))
        dt += step
    
    return pd.DataFrame(result)[0]


# In[11]:


p = range(0, 5)
d = range(1, 2)
q = range(0, 4)
ps = range(0,2)
ds = range(0,2)
qs = range(0,2)
s = range(5,24)

def build_arima_model(p,d,q,ps,ds,qs,s, df, dynamic_start_date, start_date_preds):
    
    
    train_wk = df[df.index < start_date_preds]
    test_wk = df[df.index >= start_date_preds]
    
    best_model_weekly = sm.tsa.statespace.SARIMAX(train_wk,
                                      order=(p, d, q),
                                      seasonal_order=(ps, ds, qs, s))

    best_results_weekly = best_model_weekly.fit()
    
    pred_dynamic_wk = best_results_weekly.get_prediction(start=pd.to_datetime(dynamic_start_date), dynamic=True, full_results=True)
    pred_dynamic_ci_wk = pred_dynamic_wk.conf_int()
    
    pred_wk = best_results_weekly.get_forecast(steps=60)
    
    forecasted_corn = pred_dynamic_wk.predicted_mean
    testCopy_wk = test_wk.copy()
    testCopy_wk['prices_forecasted'] = pred_wk.predicted_mean
    
    df_wkmerge = df_daily[df_daily.index >= '2021-06-01'].reset_index()
    df_wk = df_daily[df_daily.index >= '2021-12-01'].reset_index()
    first_value = df_wk['Milho B3 - D'].head(1)
    
    dict_price = {'Predicted Price':first_value}
    #new_df = df_m1.append(dict_price, ignore_index=True)

    new_df_wk = pd.DataFrame(dict_price)
    
    for v in pred_wk.predicted_mean.values:
        previous_price = new_df_wk['Predicted Price'].tail(1).values[0]
        new_price = previous_price + v
        dict_price = {'Predicted Price':new_price}
        #print(previous_price, v)
        new_df_wk = new_df_wk.append(dict_price, ignore_index=True)
    
    # generating next dates for forecast
    
    last_date = df_wk.tail(1)['Data'].dt.strftime('%Y-%m-%d').values[0].split('-')
    last_date = tuple([int(x) for x in last_date])
    
    df_final_predict_wk = pd.concat([df_wk, new_df_wk], axis=1)
    
    mse = ((df_final_predict_wk['Milho B3 - D'] - df_final_predict_wk['Predicted Price']) ** 2).mean()
    rmse = np.sqrt(mse)
    params = str((p,d,q,ps,ds,qs,s))
    result = 'The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 3))
    
    df_f1 = df_final_predict_wk[df_final_predict_wk['Data'].notnull()]
    df_f2 = df_final_predict_wk[df_final_predict_wk['Data'].isnull()].reset_index().iloc[1:,3:]
    df_f2['Data'] = generate_date_series(last_date)
    
    df_final_predict = pd.concat([df_wkmerge, df_f1, df_f2], axis=0)
    
    
    return result, round(rmse, 3), params, df_final_predict, df_wk, new_df_wk, best_results_weekly, pred_wk


# In[12]:


def get_best_params_arima(df, dynamic_start_date, start_date_preds):

    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q, ps, ds, qs, s))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], x[3], x[4], x[5], x[6]) for x in list(itertools.product(p, d, q, ps, ds, qs, s))]

    best_rmse = 4.4
    best_result = ''

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            results = build_arima_model(*param_seasonal, df, dynamic_start_date, start_date_preds)
            if results[1] < best_rmse:
                best_result = results[0]
                best_rmse = results[1]
                print(results[0] + results[2])

    print(best_result)
    


# In[13]:


ed = '2022-01-23'
df_daily_model = df_daily[df_daily.index < ed][['Milho B3 - D']]
df_daily_diff_model = df_daily_diff[df_daily_diff.index < ed][['Milho B3 - D']]


# In[14]:


# df_results = build_arima_model(3, 0, 1, 1, 1, 0, 21, df_daily_diff_model, '2021-10-24', '2021-10-30')
df_results = build_arima_model(3, 1, 3, 1, 1, 0, 21, df_daily_diff_model, '2021-11-07', '2021-11-15')


# In[15]:


def temp_autocorrection(df, correction_perc):
    df['Data'] = df['Data'].astype('str')
    df['Data'] = df['Data'].str.split(' ',expand=True)[0]
    df['Ano'] = df['Data'].str.split('-',expand=True)[0]
    df['Mes'] = df['Data'].str.split('-',expand=True)[1]
    df['Dia'] = df['Data'].str.split('-',expand=True)[2]
    df = df.groupby(['Ano','Mes']).max().reset_index()[['Data','Milho B3 - D', 'Predicted Price']]
    
    df['Difference'] = df['Predicted Price'] - df['Predicted Price'].shift(periods=1)
    
    correction_list = []
    
    for i in df['Difference'].items():
        if (i[1] > 4.5) or (i[1] < -4.5):
            correction_list.append(i[1] * (correction_perc/100))
        else:
            correction_list.append(0)
            
    
    correction_list[7] = 0
    
    df['Correction'] = correction_list
    df['Predicted Price'] = df['Predicted Price'] - df['Correction']
    
    df = df.set_index('Data')

    return df


# In[16]:


df_results[1]


# In[17]:


#df_results[3].tail(60).to_excel('Predicao Milho.xlsx', index=False)


# In[18]:


df_val = df_results[3].copy()
df_val = temp_autocorrection(df_val, 40)


# In[19]:


### Extrair para excel para o Felipe 


# In[20]:


# chart = df_results[3]
# chart = chart.set_index('Data')
# plt.plot(chart)

# ## Saving and Importing the model

# In[21]:


model_fit = df_results[6]


# In[22]:


model_fit.save('corn_sarima_18-02.pkl')