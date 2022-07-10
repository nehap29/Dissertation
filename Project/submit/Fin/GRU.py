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

data_collect = pd.read_csv("5min_data.csv")
data_collect["Date"] = data_collect["Unnamed: 0"]
data_collect.set_index("Date")
data_collect.drop("Unnamed: 0", axis=1, inplace=True)
all_data = data_collect[['Open','Open', 'High', 'Low', 'Close', 'Volume']].round(2)
all_data.head(10)
cut = int(len(all_data)*0.8)


# In[3]:


all_data['Open'].plot()


# In[4]:





# In[5]:


def train_test_splitting(all_data,time_steps,for_periods):

    training = all_data[:cut].iloc[:,0:1].values
    testing  = all_data[cut:].iloc[:,0:1].values

    x_training = []
    y_training = []
    y_training_stacked = []
    for i in range(time_steps,training_len-1): 
        x_training.append(training[i-time_steps:i,0])
        y_training.append(training[i:i+for_periods,0])
    x_training, y_training = np.array(x_training), np.array(y_training)

    x_training = np.reshape(x_training, (x_training.shape[0],x_training.shape[1],1))

    current = pd.concat((all_data["Open"][:cut], all_data["Open"][ cut :]),axis=0).values
    current = current[len(current)-len(testing) - time_steps:]
    current = current.reshape(-1,1)

    # Preparing X_test
    X_test = []
    for i in range(time_steps,testing_len+time_steps-for_periods):
        X_test.append(current[i-time_steps:i,0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    return x_training, y_training , X_test

x_training, y_training, X_test = train_test_splitting(all_data,5,2)
x_training.shape[0],x_training.shape[1]


# In[6]:


# Make the 3-D shape to a data frame so we can see: 
x_training_see = pd.DataFrame(np.reshape(x_training, (x_training.shape[0],x_training.shape[1])))
y_training_see = pd.DataFrame(y_training)
pd.concat([x_training_see,y_training_see],axis=1)


# In[7]:


# Make the 3-D shape to a data frame so we can see: 
X_test_see = pd.DataFrame(np.reshape(X_test, (X_test.shape[0],X_test.shape[1])))
pd.DataFrame(X_test_see)


# In[8]:


print("There are " + str(x_training.shape[0]) + " samples in the training data")
print("There are " + str(X_test.shape[0]) + " samples in the test data")


# In[9]:


def real_predictions_plot(preds):
    real_predictions = pd.DataFrame(columns = ['Open', 'prediction'])
    name = all_data.iloc[cut].name
    real_predictions['Open'] = all_data.loc[name:,'Open'][0:len(preds)]
    real_predictions['prediction'] = preds[:,0]

    return (real_predictions.plot() )


# In[10]:


def train_test_splitting_normalize(all_data,time_steps,for_periods):
    training = all_data[:cut].iloc[:,0:1].values
    testing  = all_data[cut:].iloc[:,0:1].values
    training_len = len(training)
    testing_len = len(testing)

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    training_scaled = sc.fit_transform(training)
    x_training = []
    y_training = []
    y_training_stacked = []
    for i in range(time_steps,training_len-1): 
        x_training.append(training_scaled[i-time_steps:i,0])
        y_training.append(training_scaled[i:i+for_periods,0])
    x_training, y_training = np.array(x_training), np.array(y_training)

    x_training = np.reshape(x_training, (x_training.shape[0],x_training.shape[1],1))

    current = pd.concat((all_data["Open"][: cut], all_data["Open"][cut :]),axis=0).values
    current = current[len(current)-len(testing) - time_steps:]
    current = current.reshape(-1,1)
    current  = sc.transform(current)
    X_test = []
    for i in range(time_steps,testing_len+time_steps-for_periods):
        X_test.append(current[i-time_steps:i,0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    return x_training, y_training , X_test, sc
x_training, y_training, X_test, sc = train_test_splitting_normalize(all_data,5,2)



# In[12]:


def GRU_model_regularization(x_training, y_training, X_test, sc):
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, GRU
    from keras.optimizers import SGD
    from keras.layers import Dropout
    
    model_parts = Sequential()
    model_parts.add(GRU(units=50, return_sequences=True, input_shape=(x_training.shape[1],1), activation='tanh'))
    model_parts.add(Dropout(0.2))
    model_parts.add(GRU(units=50, return_sequences=True, activation='tanh'))
    model_parts.add(Dropout(0.2))

    model_parts.add(GRU(units=50, return_sequences=True, activation='tanh'))
    model_parts.add(Dropout(0.2))

    model_parts.add(GRU(units=50, activation='tanh'))
    model_parts.add(Dropout(0.2))
    model_parts.add(Dense(units=1))
    model_parts.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
    model_parts.fit(x_training,y_training,epochs=50,batch_size=150, verbose=0)

    GRU_predictions = model_parts.predict(X_test)
    GRU_predictions = sc.inverse_transform(GRU_predictions)

    return model_parts, GRU_predictions

model_parts, GRU_predictions = GRU_model_regularization(x_training, y_training, X_test, sc)
GRU_predictions[1:10]
real_predictions_plot(GRU_prediction)


# In[13]:


result = GRU_prediction[:,0]   
    
f= open("5min-pred-gru.txt","w")

for i in range(0,len(result)):
    f.write(str(result[i]) +",")    

f.close()


# In[14]:


print(len(result))


# In[ ]:




