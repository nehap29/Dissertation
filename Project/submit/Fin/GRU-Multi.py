#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
np.random.seed(1)


import tensorflow
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from keras.model_setups import Sequential, load_model_setup
from keras.layers.core import Dense
from keras.layers.recurrent import GRU
from keras import optimizers


from keras.callbacks import nping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import datetime as dt
import time
plt.style.use('ggplot')


# In[2]:


data = yf.download('CL=F', start ='2000-08-23' , end='2021-04-22')
values = data.values
data.tail()

sa_news = pd.read_csv("sa_twitter.csv")
sa_news = sa_news[(9890-5155):]
print(len(data))
print(len(sa_news))

boo = sa_news["polarity"].to_list()

data["SA"] = boo

data.tail()


# In[3]:

# In[4]:

data.corr()['Close']


print(data.describe().Volume) 
data.drop(data[data['Volume']==0].index, inplace = True) 


# In[6]:

n = nping(monitor='val_loss', min_delta=0.0001, patience=80,  verbose=1, mode='min')
call = [n]


# In[7]:


#Build and train the model_setup
def fitting(train,val,step,hl,lr,batch,epochs):
    training = []
    y_training = []
    x_values = []
    y_vLUES = []
    for i in range(step,train.shape[0]):
        training.append(train[i-step:i])
        y_traininging.append(train[i][0])
    training,y_training = np.array(training),np.array(y_training)

    for i in range(step,val.shape[0]):
        x_values.append(val[i-step:i])
        y_vLUES.append(val[i][0])
    x_values,y_vLUES = np.array(x_values),np.array(y_vLUES)
    
    model_setup = Sequential()
    model_setup.add(GRU(training.shape[2],input_shape = (training.shape[1],training.shape[2]),return_sequences = True,
                   activation = 'relu'))
    for i in range(len(hl)-1):        
        model_setup.add(GRU(hl[i],activation = 'relu',return_sequences = True))
    model_setup.add(GRU(hl[-1],activation = 'relu'))
    model_setup.add(Dense(1))
    model_setup.compile(optimizer = optimizers.Adam(lr = lr), loss = 'mean_squared_error')

    history = model_setup.fit(training,y_training,epochs = epochs,batch_size = batch,validation_data = (x_values, y_vLUES),verbose = 0,
                        shuffle = False, callbacks=call)
    model_setup.reset_states()
    return model_setup, history.history['loss'], history.history['val_loss']


# In[8]:


# Evaluating the model_setup
def eval(model_setup,test,step):
    X_test = []
    Y_test = []

    # Loop for testing data
    for i in range(step,test.shape[0]):
        X_test.append(test[i-step:i])
        Y_test.append(test[i][0])
    X_test,Y_test = np.array(X_test),np.array(Y_test)
    #print(X_test.shape,Y_test.shape)
  
    # Prediction Time !!!!
    Y_hat = model_setup.predict(X_test)
    mse = mean_squared_error(Y_test,Y_hat)
    rmse = sqrt(mse)
    r = r2_score(Y_test,Y_hat)
    return mse, rmse, r, Y_test, Y_hat


# In[9]:


# Plotting the predictions
def plot_data(Y_test,Y_hat):
    plt.plot(Y_test,c = 'r')
    plt.plot(Y_hat,c = 'y')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('Prediction using Multivariate-GRU with Twitter model_setup')
    plt.legend(['Actual','Predicted'],loc = 'lower right')
    plt.show()


# In[10]:


# Plotting the training errors
def plot_error(train_loss,val_loss):
    plt.plot(train_loss,c = 'r')
    plt.plot(val_loss,c = 'b')
    plt.ylabel('Loss')
    plt.legend(['train','val'],loc = 'upper right')
    plt.show()


# In[11]:


# Extracting the series
series = data[['Close','High','SA']] # Picking the series with high correlation
print(series.shape)
print(series.tail())


# In[12]:


train_data = data[0:int(len(data)*0.6)]
valid_dinfo = data[int(len(data)*0.6): int(len(data)*0.8)]
testing_data = data[int(len(data)*0.8):]

print(train_data.shape,valid_dinfo.shape,testing_data.shape)


# In[13]:


# Normalisation
sc = MinMaxScaler()
train = sc.fit_transform(train_data)
val = sc.transform(valid_dinfo)
test = sc.transform(testing_data)
print(train.shape,val.shape,test.shape)


# In[14]:


step = 50
hl = [40,35]
lr = 1e-3
batch_size = 64
num_epochs = 250


# In[ ]:


model_setup,train_error,val_error = fitting(train,val,step,hl,lr,batch_size,num_epochs)
plot_error(train_error,val_error)


# In[ ]:


mse, rmse, r2_value,true,predicted = eval(model_setup,test,step)
print('MSE = {}'.format(mse))
print('RMSE = {}'.format(rmse))
print('R-Squared Score = {}'.format(r2_value))
plot_data(true,predicted)


# In[ ]:


f= open("twitter-pred-multi-gru.txt","w")

for i in range(0,len(predicted)):
    f.write(str(predicted[i][0]) +",")
    
f.close()

