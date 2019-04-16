# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:40:26 2019

@author: Edvan Soares
"""

import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps
		if end_ix > len(sequence)-1:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def split_train_test_data(X, y, n_test):
    return X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)

def fit_model_lstm(X, y):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=2000, verbose=0)
    return model

data = read_csv(open('data.csv', 'r'), sep='\t', header=0).values.T.tolist()[0]

n_steps = 3
n_features = 1
X, y = split_sequence(data, n_steps)
X = X.reshape((X.shape[0], X.shape[1], n_features))
X_train, y_train, X_test, y_test = split_train_test_data(X, y, 6)

model = fit_model_lstm(X_train, y_train)
predicted = []

for x in X_test:
    test = x.reshape((1, n_steps, n_features))
    yhat = model.predict(test, verbose=0)
    predicted.append(yhat)

print(mape(y_test, predicted))