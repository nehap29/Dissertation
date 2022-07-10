#!/usr/bin/env python
# coding: utf-8

# In[1]:


lstm = open("5min-pred-lstm.txt")
arima = open("pred-arima.txt")
real = open("pred-real.txt")
gru = open("5min-pred-gru.txt")
svm = open("pred-svm.txt")

a = lstm.readline()
b = arima.readline()
c = real.readline()
d = gru.readline()
e = svm.readline()

lstm = []
arima = []
real = []
gru = []
svm = []


for i in a.split(','):
    lstm.append(float(i))
    
for i in b.split(','):
    arima.append(float(i))

for i in c.split(','):
    real.append(float(i))
    
for i in d.split(','):
    gru.append(float(i))
    
for i in e.split(','):
    svm.append(float(i))
    
arima.pop()
arima.pop()

real.pop()
real.pop()

svm.pop()


print(len(lstm))
print(len(arima))
print(len(real))
print(len(gru))
print(len(svm))


# In[2]:


import yfinance as yf
data = yf.download("CL=F", period="max")
print(min(data.index))
print(max(data.index))
dates = data.index[int(len(data)*0.8):-2]


# In[3]:


import matplotlib.pyplot as plt
plt.plot(dates,lstm, label = "LSTM Prediction")
plt.plot(dates,real, label = "True Values")
plt.gcf().set_size_inches(20, 10)
plt.ylabel("Price", fontsize=18)
plt.xlabel("Date", fontsize=18)
plt.title("Comparison of LSTM predictions to true prices", fontsize=18)
plt.legend(fontsize=18)
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(dates,gru, label = "GRU Prediction")
plt.plot(dates,real, label = "True Values")
plt.gcf().set_size_inches(20, 10)
plt.ylabel("Price", fontsize=18)
plt.xlabel("Date", fontsize=18)
plt.title("Comparison of GRU predictions to true prices", fontsize=18)
plt.legend(fontsize=18)
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(dates,arima, label = "ARIMA Prediction")
plt.plot(dates,real, label = "True Values")
plt.gcf().set_size_inches(20, 10)
plt.ylabel("Price", fontsize=18)
plt.xlabel("Date", fontsize=18)
plt.title("Comparison of ARIMA predictions to true prices", fontsize=18)
plt.legend(fontsize=18)
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(dates,svm, label = "ARIMA Prediction")
plt.plot(dates,real, label = "True Values")
plt.gcf().set_size_inches(20, 10)
plt.ylabel("Price", fontsize=18)
plt.xlabel("Date", fontsize=18)
plt.title("Comparison of SVM predictions to true prices", fontsize=18)
plt.legend(fontsize=18)
plt.show()


# In[ ]:




