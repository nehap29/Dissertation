#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd

def abs_error(real_val, pred):
    return abs(real_val - pred)
    
def find_min(real_val,pred_0,pred_1,pred_2):
    res0 = abs_error(real_val, pred_0)
    res1 = abs_error(real_val, pred_1)
    res2 = abs_error(real_val, pred_2)

    if(res0 < res1 and res0 < res2):
        return 0
    elif(res1 < res0 and res1 < res2):
        return 1
    elif(res2 < res0 and res2 < res1):
        return 2
    
   
    return (-1) 
                      


# In[11]:


lstm = open("pred-lstm.txt")
arima = open("pred-arima.txt")
real = open("pred-real.txt")
gru = open("pred-gru.txt")


# In[12]:


lstm = open("pred-lstm.txt")
arima = open("pred-arima.txt")
real = open("pred-real.txt")
gru = open("pred-gru.txt")
svm = open("pred-svm.txt")
trans = open("pred-trans.txt")

a = lstm.readline()
b = arima.readline()
c = real.readline()
d = gru.readline()
e = svm.readline()
f = trans.readline()

lstm = []
arima = []
real = []
gru = []
svm = []
trans = []


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
    
for i in f.split(','):
    trans.append(float(i))
    
arima.pop()
arima.pop()

real.pop()
real.pop()

svm.pop()


lstm = lstm[(1030-968):]
arima = arima[(1030-967):]
real = real[(1030-967):]
gru = gru[(1030-967):]
svm = svm[(1030-967):]

print(len(lstm))
print(len(arima))
print(len(real))
print(len(gru))
print(len(svm))
print(len(trans))


# In[13]:


lstm_counter = 0
gru_counter = 0
arima_counter = 0
svm_counter = 0
trans_counter = 0

for i in range (0, len(real)):
    result = find_min(real[i],lstm[i],trans[i],gru[i])
    
    if (result == 0):
        lstm_counter = lstm_counter + 1
        
    elif (result == 1):
        trans_counter = trans_counter + 1
        
    elif (result == 2):
        gru_counter = gru_counter + 1
        
        
print(lstm_counter/len(real) * 100)
print(arima_counter/len(real) * 100)
print(gru_counter/len(real) * 100)
print(svm_counter/len(real) * 100)
print(trans_counter/len(real) * 100)



# In[16]:


import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'LSTM', 'GRU', 'Transformer'
sizes = [12, 65, 21]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[ ]:





# In[ ]:




