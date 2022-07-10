#!/usr/bin/env python
# coding: utf-8

# In[1]:


from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import pacf
from statsmodels.regression.linear_model import yule_walker

import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
import yfinance as yf
#AMZN = data = yf.download('CL=F', start ='2000-08-23' , end='2021-04-22')

AMZN = pd.read_csv("5min_data.csv")
AMZN["Date"] = AMZN["Unnamed: 0"]
AMZN.set_index("Date")
AMZN.drop("Unnamed: 0", axis=1, inplace=True)
all_data = AMZN[['Adj Close','Open', 'High', 'Low', 'Close', 'Volume']].round(2)
all_data.head(10)
cut = int(len(all_data)*0.8)


# In[3]:



# In[5]:

# In[6]:

xtraining_see = pd.DataFrame(np.reshape(xtraining, (xtraining.shape[0],xtraining.shape[1])))
ytraining_see = pd.DataFrame(ytraining)
pd.concat([xtraining_see,ytraining_see],axis=1)


# In[7]:


# Make the 3-D shape to a data frame so we can see: 
xtesting_see = pd.DataFrame(np.reshape(xtesting, (xtesting.shape[0],xtesting.shape[1])))
pd.DataFrame(xtesting_see)




# In[9]:


# In[10]:


def test_training_split_normalize(all_data,steps,for_periods):

    training = all_data[:cut].iloc[:,0:1].values
    testing  = all_data[cut:].iloc[:,0:1].values
    training_len = len(training)
    testing_len = len(testing)
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    training_scaled = sc.fit_transform(training)
    xtraining = []
    ytraining = []
    ytraining_stacked = []
    for i in range(steps,training_len-1): 
        xtraining.append(training_scaled[i-steps:i,0])
        ytraining.append(training_scaled[i:i+for_periods,0])
    xtraining, ytraining = np.array(xtraining), np.array(ytraining)
    xtraining = np.reshape(xtraining, (xtraining.shape[0],xtraining.shape[1],1))

    adds = pd.concat((all_data["Open"][: cut], all_data["Open"][cut :]),axis=0).values
    adds = adds[len(adds)-len(testing) - steps:]
    adds = adds.reshape(-1,1)
    adds  = sc.transform(adds)

    xtesting = []
    for i in range(steps,testing_len+steps-for_periods):
        xtesting.append(adds[i-steps:i,0])
        
    xtesting = np.array(xtesting)
    xtesting = np.reshape(xtesting, (xtesting.shape[0],xtesting.shape[1],1))

    return xtraining, ytraining , xtesting, sc

xtraining, ytraining, xtesting, sc = test_training_split_normalize(all_data,5,2)


# In[11]:

# In[12]:


def architecture(xtraining, ytraining, xtesting, sc):

    # create a model
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, LSTM
    from keras.optimizers import SGD
    from keras.layers import Dropout
    
    model_setup = Sequential()
    model_setup.add(LSTM(units=50, return_sequences=True, input_shape=(xtraining.shape[1],1), activation='tanh'))
    model_setup.add(Dropout(0.2))
    model_setup.add(LSTM(units=50, return_sequences=True, activation='tanh'))
    model_setup.add(Dropout(0.2))
    model_setup.add(LSTM(units=50, return_sequences=True, activation='tanh'))
    model_setup.add(Dropout(0.2))
    model_setup.add(LSTM(units=50, activation='tanh'))
    model_setup.add(Dropout(0.2))
    model_setup.add(Dense(units=1))
    model_setup.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
    model_setup.fit(xtraining,ytraining,epochs=50,batch_size=150, verbose=0)

    LSTM_predictions = model_setup.predict(xtesting)
    LSTM_predictions = sc.inverse_transform(LSTM_predictions)

    return model_setup, LSTM_predictions

model_setup, LSTM_predictions = architecture(xtraining, ytraining, xtesting, sc)
LSTM_predictions[1:10]
actual_pred_plot(LSTM_prediction)


# In[13]:


result = LSTM_prediction[:,0]   
    
f= open("pred-lstm.txt","w")

for i in range(0,len(result)):
    f.write(str(result[i]) +",")    

f.close()


# 