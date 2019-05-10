# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:38:15 2019

@author: I304515
"""

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
#%matplotlib inline

# load the dataset

#dataframe = read_csv('monthly_DESM_wo_noise.csv', usecols=[1], engine='python', skipfooter=3)
#dataframe = read_csv('week_DESM_wo_noise.csv', usecols=[1], engine='python', skipfooter=3)
dataframe = read_csv('data_test_kongyaji.txt', usecols=[1], engine='python')
dataframe = read_csv('raw_shuibeng_20160101_20180831_decomposed.txt', encoding='utf-8', usecols=[1])

#dataframe = read_csv('sales_weekly_raw.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
# 将整型变为floatdata_test_fadianji
dataset = dataset.astype('float32')

len(dataset)
plt.plot(dataset)
plt.show()


#diffset = numpy.diff(dataset[:,0])
diffset = dataset
diffset = diffset.reshape((len(diffset), 1))

plt.plot(diffset)
plt.show()
# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def create_dataset_test(dataset, look_back, train_size, test_size):
    dataX, dataY = [], []
    for i in range(train_size-look_back,len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for 
numpy.random.seed(7)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
normset = scaler.fit_transform(diffset)

# split into train and test sets
train_size = int(len(normset) * 0.8)
test_size = len(normset) - train_size
train, test = normset[0:train_size,:], normset[train_size:len(normset),:]

# use this function to prepare the train and test datasets for modeling
look_back = 25
trainX, trainY = create_dataset(train, look_back)
#testX, testY = create_dataset(test, look_back)
testX, testY = create_dataset_test(normset, look_back, train_size, test_size)

len(trainY)
len(testY)

#numpy.savetxt("train.csv",train)
#numpy.savetxt("trainX.csv",trainX)
#numpy.savetxt("trainY.csv",trainY)

#numpy.savetxt("testX.csv",trainX)
#numpy.savetxt("testY.csv",trainY)


# reshape input to be [samples, time steps, features]
#trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

trainX.shape
trainY.shape
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2)
model.fit(trainX, trainY, epochs=300, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
predict_array = [];
array_tmp = testX[0]
for i in range(0,len(testX)):
    array_tmp = numpy.reshape(array_tmp, (look_back, 1))
    array_tmp= numpy.array([array_tmp])
    array_tmp.ndim
    #print(array_tmp)
    predict_value = model.predict(array_tmp)
    predict_array = numpy.append(predict_array,predict_value)
    #print(predict_value)
    #array_tmp = textX[i]
    #predict_array = numpy.append(predict_array,predict_value)
    array_tmp = numpy.delete(array_tmp,[0])
    array_tmp = numpy.append(array_tmp,predict_value)
    array_tmp = numpy.array([array_tmp])

testPredict = numpy.vstack((predict_array[0],predict_array[1]))
for i in range(2,len(predict_array)):
    testPredict = numpy.vstack((testPredict,predict_array[i]))

#plt.plot(testY)
#plt.plot(testPredict)
#plt.show


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

def inverse_diff(original,dataset):
    a = []
    orig_value = original
    a.append(orig_value)
    for i in range(0,len(dataset)-1):
        orig_value = orig_value + dataset[i]
        a.append(orig_value)
    return numpy.array(a)


trainY = trainY.reshape(len(trainY[0]),1)
trainY = inverse_diff(dataset[0],trainY)
trainPredict = inverse_diff(dataset[0],trainPredict)
testY = testY.reshape(len(testY[0]),1)
testY = inverse_diff(dataset[len(dataset)-test_size],testY)
testPredict = inverse_diff(dataset[len(dataset)-test_size],testPredict)

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset+1)
testPredictPlot[:, :] = numpy.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
testPredictPlot[len(dataset)-len(testPredict):len(dataset)+1, :] = testPredict

# plot baseline and predictions
#plt.plot(scaler.inverse_transform(dataset))

plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot, color='red')
plt.show()