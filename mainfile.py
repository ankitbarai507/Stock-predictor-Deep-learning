# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:30:30 2019

@author: Ankit Barai
"""


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU,SimpleRNN
from keras.layers.core import Activation, Dropout
from keras.regularizers import l1
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg=agg.fillna(0)
    return agg
# load dataset
stock=input('enter dataset')
ticker=stock
dataset = read_csv('dataset/'+stock+'.csv', header=0, index_col=0)
if dataset is None:
    print('exit')
    exit()
'''
# reverse the dataframe
'''
dataset=dataset.iloc[::-1] 
values = dataset.values
# ensure all data is float
values = values.astype('float32')
# normalize features
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled = scaler.fit_transform(values)
# frame as supervised learning
pred_days=int(input('enter prediction days (Put 10 preferably):'))
reframed = series_to_supervised(values, 1, pred_days)
reframed.fillna(0)
# drop columns we don't want to predict
temp=8#keep it 8 to predict close price
count=0
for i in range(5, 5+(pred_days*5) ):
    if i!=temp:
        reframed.drop(reframed.columns[[i-count]], axis=1, inplace=True)
        count=count+1
    else:
        temp=temp+5
        
    
bkp=DataFrame(reframed)
print(reframed.head())
scaler = MinMaxScaler(feature_range=(0, 1))
 

reframed = scaler.fit_transform(reframed)
# split into train and test sets
reframed=DataFrame(reframed)
values = reframed.values
n_train_days = int(0.80*len(values))
train = values[1:n_train_days, :]

test = values[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, 0:5], train[:,5:]
test_X, test_y = test[:, 0:5], test[:, 5:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

def get_LSTMmodel(): 
    # design network
    model = Sequential()#50 was first arg of below LSTM() statement
    model.add(LSTM(54,input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128,return_sequences=False))#128 is number of neurons ,activity_regularizer=l1(0.001)
    model.add(Dropout(0.2))
    model.add(Dense(pred_days,activation='linear'))#,activity_regularizer=l1(0.001)
  
   
    #model.add(Activation('linear'))
    model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
    return model
    #model.save_weights('model.h5')
    #initial_weights = model.get_weights()

def get_GRUmodel(): 
    # design network
    model = Sequential()#50 was first arg of below LSTM() statement
    model.add(GRU(54,input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(128,return_sequences=False))#128 is number of neurons ,activity_regularizer=l1(0.001)
    model.add(Dropout(0.2))
    model.add(Dense(pred_days,activation='linear'))#,activity_regularizer=l1(0.001)

    #model.add(Activation('linear'))
    model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
    return model
    #model.save_weights('model.h5')
    #initial_weights = model.get_weights()

def get_RNNmodel(): 
    # design network
    model = Sequential()#50 was first arg of below LSTM() statement
    model.add(SimpleRNN(54,input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(128,return_sequences=False))#128 is number of neurons ,activity_regularizer=l1(0.001)
    model.add(Dropout(0.2))
    model.add(Dense(pred_days,activation='linear'))#,activity_regularizer=l1(0.001)

    #model.add(Activation('linear'))
    model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
    return model
    #model.save_weights('model.h5')
    #initial_weights = model.get_weights()

def get_CNNmodel():
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(pred_days))
    model.compile(loss='mse', optimizer='adam')
    return model

#
# fit network
epoch=int(input('enter number of epochs '))
batch_size=int(input('enter batch size or window size'))
ch=int(input('enter 1-GRU,2-RNN,3-LSTM,4-CNN , CNN not works'))
if ch == 1:
    stock+='GRU_'
    model1=get_GRUmodel()
elif ch == 2:
    stock+='RNN_'
    model1=get_RNNmodel()
elif ch==3:
    stock+='LSTM_'
    model1=get_LSTMmodel()
elif ch ==4:
    stock+='CNN_'
    model1=get_CNNmodel()
#model1=get_RNNmodel()
history = model1.fit(train_X, train_y, epochs=epoch, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.xlabel('Number of Iterartions')
pyplot.ylabel('Loss')
pyplot.savefig('output/'+ticker+'/'+stock+'_losscurve_'+str(epoch)+'epoch'+str(batch_size)+'batchsize'+'.png')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = model1.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_X[:,0:5],yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,5:]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), pred_days))
inv_y = concatenate(( test_X[:, 0:5],test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,5:]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y[0], inv_yhat[0]))
print('Test RMSE: %.3f' % rmse)

print('MAPE',mean_absolute_percentage_error(inv_y[0], inv_yhat[0]))

pyplot.plot([k[0] for k in inv_y],label='actual')
pyplot.plot([k[0] for k in inv_yhat], label='predicted')
pyplot.xlabel('Trading days')
pyplot.ylabel('INR')
pyplot.savefig('output/'+ticker+'/'+stock+'_prediction_'+str(epoch)+'epoch'+str(batch_size)+'batchsize'+'.png')
pyplot.legend()
pyplot.show()
test_X = test_X.reshape((test_X.shape[0], 1,test_X.shape[1]))
#test_y = test_y.reshape((test_y.shape[0], 1, test_y.shape[1]))
score, acc = model1.evaluate(test_X, test_y,batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
'''
pyplot.plot(inv_y,label='actual')
pyplot.plot(inv_yhat, label='predicted')
pyplot.legend()
pyplot.show()
'''
#model=model.load_weights('model.h5')
#model.reset_states()

'''
str1='SBI.csv'
from pandas import read_csv
from matplotlib import pyplot
# load dataset
dataset = read_csv(str1, header=0,index_col=0)
'''
# reverse the dataframe
'''
dataset=dataset.iloc[::-1] 
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 4]
i = 1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()
'''