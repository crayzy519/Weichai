# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:37:33 2019

@author: I304515
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:35:18 2019

@author: I304515
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:33:28 2019

@author: I304515
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt

from pandas import Series

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

os.getcwd()
os.chdir("c:/ai_workspace")

df = pd.read_csv('raw_shuibeng_20160101_20180831.txt', encoding='utf-8')

ts = df['VALUE']  # 生成pd.Series对象
ts.plot()

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts, model="additive", freq = 7)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

trend.plot()
seasonal.plot()
residual.plot()

# 移动平均图
def draw_trend(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.ewma(timeSeries, span=size)

    timeSeries.plot(color='blue', label='Original')
    rolmean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()

def draw_ts(timeSeries):
    f = plt.figure(facecolor='white')
    timeSeries.plot(color='blue')
    plt.show()

'''
　　Unit Root Test
   The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
   root, with the alternative that there is no unit root. That is to say the
   bigger the p-value the more reason we assert that there is a unit root
'''
def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

# 自相关和偏相关图，默认阶数为31阶
def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.show()

# test stationary directly
draw_ts(trend)
trend.dropna(inplace=True)
testStationarity(trend)
# diff 1 stationary
trend_diff1 = trend.diff(1)
trend_diff1.dropna(inplace=True)
testStationarity(trend_diff1)
draw_acf_pacf(trend_diff1)
draw_ts(trend_diff1)
# log stationary with diff1
trend_log = np.log(trend)
draw_ts(trend_log)
trend_log_diff1 = trend_log.diff(1)
trend_log_diff1.dropna(inplace=True)
testStationarity(trend_log_diff1)
draw_acf_pacf(trend_log_diff1)
draw_ts(trend_log_diff1)

#divide into train and validation set
train_trend = trend_diff1[:int(0.8*(len(trend_diff1)))]
valid_trend = trend_diff1[int(0.8*(len(trend_diff1))):]

from pmdarima.arima import auto_arima
model_trend = auto_arima(train_trend, start_p=1, start_q=1,
                          max_p=10, max_q=10, m=55,
                          start_P=0, seasonal=True,
                          d=1, D=1, trace=True,
                          error_action='ignore',  
                          suppress_warnings=True, 
                          stepwise=True
                          ); 
                         
model_trend.fit(train_trend)

forecast_trend = model_trend.predict(n_periods=len(valid_trend))
forecast_trend = pd.DataFrame(forecast_trend,index = valid_trend.index,columns=['Prediction'])

#plot the predictions for validation set
plt.plot(train_trend, label='Train')
plt.plot(valid_trend, label='Valid')
plt.plot(forecast_trend, label='Prediction')
plt.show()

#handling seasonal data
draw_ts(seasonal)
seasonal.dropna(inplace=True)
testStationarity(trend)

seasonal_diff1 = seasonal.diff(1)
seasonal_diff1.dropna(inplace=True)
testStationarity(seasonal_diff1)
draw_ts(seasonal_diff1)


#divide into train and validation set
train_seasonal = seasonal[:int(0.8*(len(seasonal)))]
valid_seasonal = seasonal[int(0.8*(len(seasonal))):]

from pmdarima.arima import auto_arima
model_seasonal = auto_arima(train_seasonal, start_p=1, start_q=1,
                          max_p=10, max_q=10, m=7,
                          start_P=0, seasonal=False,
                          d=1, D=1, trace=True,
                          error_action='ignore',  
                          suppress_warnings=True, 
                          stepwise=True
                          ); 

model_seasonal.fit(train_seasonal)

forecast_seasonal = model_seasonal.predict(n_periods=len(valid_seasonal))
forecast_seasonal = pd.DataFrame(forecast_seasonal,index = valid_seasonal.index,columns=['Prediction'])

#plot the predictions for validation set
plt.plot(train_seasonal, label='Train')
plt.plot(valid_seasonal, label='Valid')
plt.plot(forecast_seasonal, label='Prediction')
plt.show()


draw_ts(residual)
residual.dropna(inplace=True)
testStationarity(residual)

#divide into train and validation set
train_residual = residual[:int(0.8*(len(residual)))]
valid_residual = residual[int(0.8*(len(residual))):]

from pmdarima.arima import auto_arima
model_residual = auto_arima(train_residual, start_p=1, start_q=1,
                          max_p=10, max_q=10, m=55,
                          start_P=0, seasonal=True,
                          d=1, D=1, trace=True,
                          error_action='ignore',  
                          suppress_warnings=True, 
                          stepwise=True
                          ); 

model_residual.fit(train_residual)

forecast_residual = model_residual.predict(n_periods=len(valid_residual))
forecast_residual = pd.DataFrame(forecast_residual,index = valid_residual.index,columns=['Prediction'])

#plot the predictions for validation set
plt.plot(train_residual, label='Train')
plt.plot(valid_residual, label='Valid')
plt.plot(forecast_residual, label='Prediction')
plt.show()

def inverse_diff(original,dataset):
    a = []
    orig_value = original
    first_idx = dataset.head(1).index.values[0]
    #a.append(orig_value)
    for i in range(first_idx, first_idx+len(dataset)):
        orig_value = orig_value + dataset[i]
        a.append(orig_value)
    
    df1 = pd.DataFrame(a)
    df1.index = pd.RangeIndex(start=first_idx, stop=first_idx+len(dataset), step=1)
    return df1

valid_trend_inv = inverse_diff(trend[valid_trend.head(1).index-1].values[0],valid_trend)
forecast_trend_inv = inverse_diff(trend[forecast_trend.head(1).index-1].values[0],forecast_trend['Prediction'])
forecast_trend_inv.columns = ['Prediction']


valid = valid_residual + valid_seasonal + valid_trend_inv[0]
forecast = forecast_residual + forecast_seasonal + forecast_trend_inv

valid.dropna(inplace=True)
forecast.dropna(inplace=True)

plt.plot(valid)
plt.plot(forecast)
plt.show()

#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error
rms = sqrt(mean_squared_error(valid,forecast))
print(rms)

rms = sqrt(mean_squared_error(valid[0:8],forecast[0:8]))
print(rms)

