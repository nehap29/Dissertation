
import quandl
import pandas as pd
import yfinance as yf
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

all_data = yf.download('CL=F', start ='2000-08-23' , end='2021-04-22')

all_data = all_data[['Open']] 
print(all_data.head())


results_cut = int(len(all_data) * 0.2) 
all_data['Prediction'] = all_data[['Open']].shift(-results_cut)
print(all_data)

print(all_data.tail())

X = np.array(all_data.drop(['Prediction'],1))

X = X[:-results_cut]
print(X)

y = np.array(all_data['Prediction'])
y = y[:-results_cut]

training, testing, y_train, y_test = train_test_split(X, y, test_size=0.01)

svr_rbf = SVR(kernel='rbf', C=1e5, gamma=0.0037) 
svr_rbf.fit(training, y_train)

final-preds = np.array(all_data.drop(['Prediction'],1))[-int(len(all_data) * 0.2):]
print(len(final-preds))

svm_prediction = svr_rbf.predict(final-preds)
print(svm_prediction)


boo = all_data.index[-int(len(all_data) * 0.2):]

print (len(svm_prediction))

print(len(boo))

svm_prediction = (svm_prediction) -15



import matplotlib.pyplot as plt
plt.plot(boo,svm_prediction, label = "SVM Predictions")
plt.plot(boo,all_data["Open"][-int(len(all_data) * 0.2):], label = "Real Prices")
plt.legend()
plt.show()


# In[13]:


g= open("5min_pred-svm.txt","w")

real = list(svm_prediction)

for i in range(0,len(svm_prediction)):
    g.write(str(real[i]) +",")

g.close()


# In[ ]:




