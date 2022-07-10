#!/usr/bin/env python
# coding: utf-


# In[2]:


# Import Libraries

from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
nltk.download('vader_lexicon')
import pycountry
import re
import string

from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer


# In[3]:


# Authentication
consumerKey = 'JvaUYSYyckryhmmCNQLp4DuK6'
consumerSecret = '3p07FNUAKUJMpg9ANpCgEBrVAqw8H4dcAVACjWO1FZEDBsNsg9'
accessToken = '1276677741799186432-bxjvVd7dX4wk4P7cegysfHi1fYFbYu'
accessTokenSecret = 'XAAKBrOQMw0CavFPTeWptsv8un1xrVNhVgDc1dBKCBqWB'

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth, wait_on_rate_limit=True)


# In[4]:


#Sentiment Analysis

def percentage(part,whole):
    return 100 * float(part)/float(whole) 

word = input()
num = int(input ())


tweets = tweepy.Cursor(api.search, q=word).items(num)

all_data = []
dates = []

for tweet in tweets:
    
    #print(tweet.text)
    dates.append(tweet.created_at)
    all_data.append(tweet.text)
    analysis = TextBlob(tweet.text)
    score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
    comp = score['compound']
    polarity += analysis.sentiment.polarity


polarity = percentage(polarity, num)

print(dates)


# In[5]:

store = pd.DataFrame(all_data)
store['Dates'] = dates
store.drop_duplicates(inplace = True)
store["text"] = store[0]
remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
rt = lambda x: re.sub("(@[A-Za-z0–9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
store["text"] = store.text.map(remove_rt).map(rt)
store["text"] = store.text.str.lower()

store.head(10)


store[['polarity', 'subjectivity']] = store['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in store['text'].iteritems():
 score = SentimentIntensityAnalyzer().polarity_scores(row)
    
store.tail(10)


# In[7]:

store.tail()


# In[10]:


store.drop(["text","subjectivity","sentiment"], axis=1, inplace=True)
store.tail()


# In[11]:


data = pd.read_csv("tweets_on_crude_oil.csv")
data.tail()


# In[12]:


data.drop(["0", "Unnamed: 0", "text","subjectivity","sentiment", "neg", "neu", "pos", "compound"],axis=1,inplace=True)


# In[13]:


data.head()
dates = data["Dates"].to_list()
print(len(data))
import pandas as pd
date1 = '1994-03-18'
date2 = '2021-04-14'
mydates = pd.date_range(date1, date2).tolist()

data.index = mydates


# In[14]:


data.index = pd.DatetimeIndex(data.index)
data = data[~data.index.duplicated(keep='first')]


# In[15]:


idx = pd.date_range(min(data.index),max(data.index))

data = data.reindex(idx, fill_value=0)

print(data)


# In[22]:


import matplotlib.pyplot as plt
plt.plot(dates,data["polarity"])

plt.gcf().set_size_inches(20, 10)
plt.xticks(np.arange(0,len(dates), 1500), dates[0:len(dates):1500])
plt.ylabel("Polarity", fontsize=18)
plt.xlabel("Date", fontsize=18)
plt.title("Sentiment of Tweets", fontsize=18)

plt.show()


# In[17]:


data.tail()
#dat.to_csv('sa_news.csv')


# In[23]:


print(data.head())


# In[24]:


data.drop("Dates",axis=1,inplace=True)


# In[25]:


print(data)


# In[26]:


data.to_csv('sa_twitter.csv')


# In[ ]:




