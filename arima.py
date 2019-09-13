# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 10:09:40 2019
Reference :https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
@author: ACER

ARIMA model

"""


from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import statistics
def parser(x):
    return datetime.strptime(x,'%Y-%m-%d')

series = read_csv(input('enter dataset')+'.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series = series.dropna()
series = series.fillna(0)
for col in series.columns:
    if col!='Close':
        series=series.drop(col,1)
X = series.values

size = int(len(X)-10)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
pe=[]
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    if t!=3:
        pe.append(float(abs(yhat-obs)/obs)*100)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
print('mape=',statistics.mean(pe))
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


'''
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')

series = read_csv('NSE_SBI.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# fit model
for col in series.columns:
    if col!='Close':
        series=series.drop(col,1)
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())
'''