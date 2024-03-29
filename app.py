import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.graph_objs.scatter import Line

from statsmodels.tsa.statespace.sarimax import SARIMAXResults

import data_transform as dt

from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta

import zipfile
with zipfile.ZipFile('corn_sarima_18-02.zip', 'r') as zip_ref:
    zip_ref.extractall()

@st.cache
def get_df(filename):
    df = pd.read_csv(filename)[['Data','Último']]
    df.rename(columns={'Último':filename.split('.')[0]}, inplace=True)
    df['Data'] = pd.to_datetime(df['Data'], format='%d-%m-%Y')
    df = df.set_index('Data')
    df = df.asfreq(freq='W', method='bfill')
    return df

filename = 'Milho B3 - D.csv'
model_filename = 'corn_sarima_18-02.sav'
model = SARIMAXResults.load(model_filename)
today = date.today().strftime("%d-%m-%Y")

df = get_df(filename)

df_diff = dt.transform_diff(df)
end_date = '2022-01-23'

df_daily = df.asfreq(freq='W', method='bfill')
df_daily_model = df[df.index < end_date][['Milho B3 - D']]
df_daily_diff_model = df_diff[df_diff.index < end_date][['Milho B3 - D']]



def loaded_results(model, df, dynamic_start_date, start_date_preds):
    
    #train_wk = df[df.index < start_date_preds]
    test_wk = df[df.index >= start_date_preds]
    
    #pred_dynamic_wk = model.get_prediction(start=pd.to_datetime(dynamic_start_date), dynamic=True, full_results=True)
    #pred_dynamic_ci_wk = pred_dynamic_wk.conf_int()
    
    pred_wk = model.get_forecast(steps=80)
    
    #forecasted_corn = pred_dynamic_wk.predicted_mean
    testCopy_wk = test_wk.copy()
    testCopy_wk['prices_forecasted'] = pred_wk.predicted_mean

    df_wkmerge = df_daily[df_daily.index >= '2021-09-01'].reset_index()
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
    #params = str((p,d,q,ps,ds,qs,s))
    #result = 'The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 3))
    
    df_f1 = df_final_predict_wk[df_final_predict_wk['Data'].notnull()]
    df_f2 = df_final_predict_wk[df_final_predict_wk['Data'].isnull()].reset_index().iloc[1:,3:]
    df_f2['Data'] = dt.generate_date_series(last_date)
    
    df_final_predict = pd.concat([df_wkmerge, df_f1, df_f2], axis=0)
    
    return df_final_predict, mse, rmse

def get_next_price(df_predict):
    now = date.today().strftime("%Y-%m-%d")
    next_price = df_predict[df_predict.index > now]['Predicted Price'][0]
    return next_price.round(2)

def get_future_prices_periods(df, months_interval):
    
    date_format = '%d-%m-%Y'
    d1 = date.today().strftime("%d-%m-%Y")
    d1 = datetime.strptime(d1, date_format)
    future_date = (d1 + relativedelta(months=months_interval)).strftime("%Y-%m-%d")
    future_price = df[df.index > future_date].head(1)['Predicted Price'].values[0].round(2)
    
    return future_price

def get_delta_interval(start_date, interval):
    date_format = '%d-%m-%Y'
    d1 = start_date
    d1 = datetime.strptime(d1, date_format)
    delta = (d1 + relativedelta(months=interval))
    return delta

def calculate_delta(x, y):
    delta = ((y - x)/x) * 100
    return str(delta.round(2)) + '%'

    
results = loaded_results(model, df_daily_diff_model, '2021-11-07', '2021-11-15')
df_val_autocorrected = dt.temp_autocorrection(results[0], 0, results[2])

df_daily_ts = dt.transform_reset_indexes(df_daily)


def groupby_dfs(df):
    df = df[df.index > get_delta_interval(today, -6)].reset_index()
    df['Data'] = df['Data'].astype('str')
    df['Data'] = df['Data'].str.split(' ',expand=True)[0]
    df['Ano'] = df['Data'].str.split('-',expand=True)[0]
    df['Mes'] = df['Data'].str.split('-',expand=True)[1]

    df = df.sort_values('Data').groupby(['Ano','Mes']).head(1)[['Data','Milho B3 - D']]
    df = df.set_index('Data')
    #df = df.groupby(['Ano','Mes']).max().reset_index()[['Data','Milho B3 - D']]
    
    return df

df_daily_ts = groupby_dfs(df_daily_ts)

#df_daily_ts = df_daily_ts.groupby('Milho B3 - D').max()

df_val_loaded_ts = dt.transform_reset_indexes(df_val_autocorrected)

st.sidebar.header("Model Assumptions")

with st.sidebar.form(key='inputs_form'):
    pass

st.title("Demo - Forecasting Milho B3")

#KPIs
nxt_price = get_next_price(df_val_autocorrected)
nxt_price_3m = get_future_prices_periods(df_val_autocorrected,2)
nxt_price_6m = get_future_prices_periods(df_val_autocorrected,5)
nxt_price_12m = get_future_prices_periods(df_val_autocorrected,11)

col1, col2, col3, col4 = st.columns(4)
col1.metric('Próx Preço', 'R$: ' + str(nxt_price))
col2.metric("Preço 3 Meses", 'R$: ' + str(nxt_price_3m), delta=calculate_delta(nxt_price, nxt_price_3m))
col3.metric("Preço 6 Meses", 'R$: ' + str(nxt_price_6m), delta=calculate_delta(nxt_price, nxt_price_6m))
col4.metric("Preço 12 Meses", 'R$: ' + str(nxt_price_12m), delta=calculate_delta(nxt_price, nxt_price_12m))

print(df_val_autocorrected)

st.text(' *Variação de preços em relação ao próximo preço')
#col5.metric('Erro Médio²', results[2].round(2))


st_fig = go.Figure([
    go.Scatter(
        x = df_daily_ts.index,
        y = df_daily_ts['Milho B3 - D'],
        mode='lines+markers',
        name='Preço Executado',
    ),
    go.Scatter(
        name='Preço Médio',
        x=df_val_autocorrected.index,
        y=df_val_autocorrected['Predicted Price'],
        mode='lines+markers',  
    ),
    go.Scatter(
        name='Máxima',
        x=df_val_autocorrected.index,
        y=df_val_autocorrected['Predicted Upper'],
        mode='lines',
        marker=dict(color="#333"),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Mínima',
        x=df_val_autocorrected.index,
        y=df_val_autocorrected['Predicted Lower'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    ),
])


st_fig.update_traces(textposition="bottom right")

#fig3 = go.Figure(data=fig1.data,
#                 layout={'xaxis':{'title':'Data'},
#                         'yaxis':{'title':'Preço em R$'}})

st.header("Forecasting Preço do Milho em R$")
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)
#layoutPick = st.radio('Graphing library',['Plotly','Matplotlib'])
#if layoutPick == 'Plotly': 
st_fig.update_layout(
    autosize=False,
    width=800,
    height=500,
    yaxis_title='Preço do Milho em R$',
    hovermode="x"
    )
st.plotly_chart(st_fig, use_container_width=False)
