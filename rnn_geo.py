## Data Preprocessing
import numpy as np #allow us to do array manipulation
import matplotlib.pyplot as plt #to visualize the data
import pandas as pd #to import and manage the dataset

## Importing the training set
# importing as dataframe using pandas
dataset_train = pd.read_csv('Google_Stock_Price_Train_2013-2017.csv');
# Select the required column using iloc method and .values converts dataframe to array
# .iloc[all_columns: only 1 row (open stock)]
# convert the dataset to numpy array as neural network accepts only arrays as inputs
training_set = dataset_train.iloc[:,1:2].values # should give as range if we give [:,1] we just get a vector what we need is numpy array

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) # to get all stock price between 0 & 1
# apply this sc object on our data
training_set_scaled = sc.fit_transform(training_set)

## Creating data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(60, 1260): # 60 inputs till the total range of the dataset
    X_train.append(training_set_scaled[i-60:i, 0]) # 60-60 = 0 so 0 to 60 indexes ,0 is for the column(we have only 1 column now :P)
    y_train.append(training_set_scaled[i, 0]) # index starts at 0 :P
    
#converting X_train and Y_train to numpy array sinse they are now a list
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping - to add additional dimentions
# (batch_size, timesteps, input_dim)
# batch_size : total number of stock prices #X_train.shape[0] = gets the total rows 
# timesteps which is 60 # X_train.shape[0] = gets the total columns
# input_dim = 1 since we are using only 1 indicator
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #(batch_size, timesteps, input_dim)


## Building the RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising th RNN
# Classification is for predicting a category or a cass
# Regression is for predicting a continuous value
regressor = Sequential()

# Adding the first LSTM layer and some dropout reqularisation (to avoid overfitting)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# for the dropout
regressor.add(Dropout(0.2)) # 20% dropout - neurons in LSTM will be ignored in each iteration of the training

#second layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#third layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#forth layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


## Adding the Output Layer
regressor.add(Dense(units = 1))


## Compiling the RNN with loss function
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

## Fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)




## P3 - Making the prediction and visualising the results

# Getting the real stock price of 2018
dataset_test = pd.read_csv('Google_Stock_Price_Test_2018-2018.csv');
real_stock_price = dataset_test.iloc[:,1:2].values

# Getting the predicted stock price
#concatenating both the train and test sets
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0); # 1 for horizontal concatenation & 0 for vertical
# getting the new inputs fo each financial day
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
    
X_test = np.array(X_test)
# for the 3D structure
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price);


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

