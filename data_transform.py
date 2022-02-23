import datetime
import pandas as pd

from statsmodels.tsa.stattools import adfuller, acf, pacf

def transform_diff(df):
    df_diff = df - df.shift(periods=1)
    df_diff.dropna(inplace=True)
    df_diff = df_diff.asfreq(freq='W', method='bfill')
    return df_diff

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
        if (i[1] > 4) or (i[1] < -4):
            correction_list.append(i[1] * (correction_perc/100))
        else:
            correction_list.append(0)
            
    
    correction_list[7] = 0
    
    df['Correction'] = correction_list
    df['Predicted Price'] = df['Predicted Price'] - df['Correction']
    
    df = df.set_index('Data')

    return df
    
def transform_reset_indexes(df):
    
    df = df.reset_index()
    df['Data'] = df['Data'].astype('str')
    df['Data'] = pd.to_datetime(df.Data, format='%Y-%m-%d')
    df = df.set_index('Data')

    return df