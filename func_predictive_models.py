import itertools
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

#Function for making a time series on a province and plotting the rolled mean and standard deviation
def roll(ts_data, region, column='total_cases'):
    test_s=ts_data.loc[(ts_data['province']==region)]  
    test_s=test_s[['date',column]]
    test_s=test_s.set_index('date')
    test_s.astype('int64')
    a=len(test_s.loc[(test_s[column]>=10)])
    test_s=test_s[-a:]
    return (test_s.rolling(window=7,center=False).mean().dropna())

def split(ts, forecast_days=15):
    #size = int(len(ts) * math.log(0.80))
    size=-forecast_days
    train= ts[:size]
    test = ts[size:]
    return(train,test)

def mape(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean(np.abs((y1 - y_pred) / y1)) * 100

def arima_province(ts_data, province, column='total_cases', forecast_days=15):
    rolling = roll(ts_data, province, column)
    train, test = split(rolling.values, forecast_days)
    p=d=q=range(0,7)
    a=99999
    pdq=list(itertools.product(p,d,q))
    
    #Determining the best parameters
    for var in pdq:
        try:
            model = ARIMA(train, order=var)
            result = model.fit(disp=0)
            if (result.aic<=a) :
                a=result.aic
                param=var
        except:
            continue
        
    #Modeling
    model = ARIMA(train, order=param)
    result = model.fit(disp=0)
    
    pred=result.forecast(steps=len(test))[0]
    #Printing the error metrics
    model_error = mape(test,pred)
    #Plotting results
    #fig = go.Figure()
    fig = None

    return (pred, result, fig, model_error, rolling.index, rolling[column])


import plotly.graph_objects as go

def arima_chart(province, arima_provinces_df):
    arima_filtered = arima_provinces_df.loc[arima_provinces_df.province == province]
    fig_arima = go.Figure()
    fig_arima.add_trace(go.Scatter(x=arima_filtered['date'][:-15], y=arima_filtered['value'][:-15], name='Historical data', mode='lines'))
    fig_arima.add_trace(go.Scatter(x=arima_filtered['date'][-15:], y=arima_filtered['value'][-15:], name='Validation data', mode='lines'))
    fig_arima.add_trace(go.Scatter(x=arima_filtered['date'][-15:], y=arima_filtered['pred'][-15:], name='Forecast', mode='lines'))
    fig_arima.update_layout(title = f'True vs Predicted values for total cases (7 days rolling mean) in {province} for 15 days', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    arima_error = arima_filtered['error'].values[0]
    return fig_arima, arima_error


from statsmodels.tsa.holtwinters import Holt

def double_exp_smoothing(ts_data, province, column='total_cases', forecast_days=15):
    df = ts_data.set_index('date')
    df = df.loc[df['province'] == province]
    df = df.resample("D").sum()
    
    train = df.iloc[:-forecast_days]
    test = df.iloc[-forecast_days:]
    pred = test.copy()
    
    model = Holt(np.asarray(train[column].values))
    model._index = pd.to_datetime(train.index)

    fit1 = model.fit(smoothing_level=.3, smoothing_trend=.05)
    pred1 = fit1.forecast(15)
    fit2 = model.fit(optimized=True)
    pred2 = fit2.forecast(15)
    fit3 = model.fit(smoothing_level=.3, smoothing_trend=.2)
    pred3 = fit3.forecast(15)

    fig_exp_smoothing_double = go.Figure()
    fig_exp_smoothing_double.add_trace(go.Scatter(x=train.index, y=train[column], name='Historical data', mode='lines'))
    fig_exp_smoothing_double.add_trace(go.Scatter(x=test.index, y=test[column], name='Validation data', mode='lines', marker_color='coral'))

    for p, f, c in zip((pred1, pred2, pred3),(fit1, fit2, fit3),('darkcyan','gold','cyan')):
        fig_exp_smoothing_double.add_trace(go.Scatter(x=train.index, y=f.fittedvalues, marker_color=c, mode='lines',
                                name=f"alpha={str(f.params['smoothing_level'])[:4]}, beta={str(f.params['smoothing_trend'])[:4]}")
        )
        fig_exp_smoothing_double.add_trace(go.Scatter(
            x=pd.date_range(start=test.index.min(), periods=len(test) + len(p)),
            y=p, marker_color=c, mode='lines', showlegend=False)
        )
        print(f"\nMean absolute percentage error: {mape(test[column].values,p).round(2)} (alpha={str(f.params['smoothing_level'])[:4]}, beta={str(f.params['smoothing_trend'])[:4]})")

    fig_exp_smoothing_double.update_layout(title=f"Holt (double) exponential smoothing for {'new cases' if column == 'new_cases' else 'total cases'} in {province}")
    return fig_exp_smoothing_double


from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from datetime import timedelta

def triple_exp_smoothing(ts_data, province, column='total_cases', forecast_days=15, validation=False):
    """
    Inputs:
        ts_data = time series pandas dataframe. Columns: date, province, variable
        province = province name to filter the time series data by
        column = variable column to make the predictions for
        forecast_days = number of days to make predictions for
        validation = boolean indicating whether to draw validation data line on the chart
        
    Outputs:
        1) a plotly figure containing:
            - a line for training data
            - a line for model fitted values
            - a line for forecast
            - [optional] a line for validation data
        2) mean absolute percentage error (float)
            calculated between testing and forecast data, if validation is set to True
            calculated between training data and model fitted values, if validation is set to False
    """
        
    df = ts_data.set_index('date')
    # replace zeros with 0.1 as the multiplicative seasonal element o HWES requires strictly positive values
    df = df.loc[((df['province'] == province) & (df[column].notnull()))].replace(0,1)
    df = df.resample("D").sum()
    
    if validation == True:
        train = df.iloc[:-forecast_days]
        test = df.iloc[-forecast_days:]
    else:
        train = df.copy()#.iloc[30:]
        test = df.copy()#.iloc[30:]

    model_triple = HWES(train[column], seasonal_periods=14, trend='add', seasonal='mul')
    fitted_triple = model_triple.fit(optimized=True, use_brute=True)
    pred_triple = fitted_triple.forecast(steps=forecast_days)
    
    if validation == True:
        pred_triple_error = mape(test[column].values,pred_triple).round(2)
    else:
        pred_triple_error = mape(train[column].values, fitted_triple.fittedvalues).round(2)
    
    #print(f"\nMean absolute percentage error: {pred_triple_error}")

    #plot the training data, the test data and the forecast on the same plot
    fig_exp_smoothing_triple = go.Figure()
    fig_exp_smoothing_triple.add_trace(go.Scatter(
        x=train.index[30:], y=train[column][30:],
        name='Historical data', mode='lines'))
    fig_exp_smoothing_triple.add_trace(go.Scatter(
        x=train.index[30:], y=fitted_triple.fittedvalues[30:],
        name='Model fit', mode='lines', marker_color='lime'))
    
    if validation == True:
        fig_exp_smoothing_triple.add_trace(go.Scatter(
            x=test.index, y=test[column],
            name='Validation data',
            mode='lines', marker_color='coral')
        )
    
    fig_exp_smoothing_triple.add_trace(go.Scatter(
        x=pd.date_range(start=train.index.max() + timedelta(1), periods=len(test) + len(pred_triple)),
        y=pred_triple, name='Forecast', marker_color='gold', mode='lines')
    )
    fig_exp_smoothing_triple.update_layout(title=f'Holt-Winters (triple) exponential smoothing for {"new cases" if column == "new_cases" else "total cases" if column == "total_cases" else "reproduction number"} in {province} for {forecast_days} days')
    return fig_exp_smoothing_triple, pred_triple_error
