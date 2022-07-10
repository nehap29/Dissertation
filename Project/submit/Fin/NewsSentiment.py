#!/usr/bin/env python
# coding: utf-8

# In[1]:


import bs4
from datetime import datetime
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
import pandas as pd
from selenium import webdriver                    # Import module 
from selenium.webdriver.common.keys import Keys   # For keyboard keys 
import time                                       # Waiting function  
import matplotlib.pyplot as plt

news_url="https://news.google.com/news/rss/search?q=crude+oil&num=400"
Client=urlopen(news_url)
xml_page=Client.read()
Client.close()

soup_page=soup(xml_page,"xml")
news_list=soup_page.findAll("item")
print(len(news_list))
# Print news title, url and publish date

titles = []
dates = []

for news in news_list:
  titles.append(news.title.text)
  dates.append(datetime.strptime(news.pubDate.text, '%a, %d %b %Y %H:%M:%S %Z'))


# In[2]:


browser = webdriver.Safari() 
for page in range (0,10):
    browser.get('https://www.businesstimes.com.sg/search/crude%20oil?page=' + str(page))       # 1 
    time.sleep(5) 
    # 2 
    data = browser.find_elements_by_class_name("media-body") 

    for i in range (0, len(data)):
        titles.append((data[i].find_element_by_tag_name("a")).text)
        dates.append(datetime.strptime(((data[i].find_element_by_tag_name("time")).text),'%d %b %Y'))

df = pd.DataFrame({'Date':dates,'Titles':titles})

print(df.tail)


# In[3]:


# Import Libraries

from textblob import TextBlob
import sys
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


def percentage(part,whole):
    return 100 * float(part)/float(whole) 

positive  = 0
negative = 0
neutral = 0
polarity = 0
data_all = []


# In[5]:
sentiment = df
sentiment["text"] = sentiment["Titles"]


# In[6]:

sentiment[['polarity', 'subjectivity']] = sentiment['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in sentiment['text'].iteritems():
 score = SentimentIntensityAnalyzer().polarity_scores(row)
 comp = score['compound']
 
sentiment.head(10)



# In[10]:


df.plot(x='Date', y='polarity')

#df.drop(['Titles', 'text', 'subjectivity', 'subjectivity' , 'sent' ], axis=1)


# In[11]:


df.drop(['Titles','text','subjectivity','sentiment'] , axis=1, inplace=True)
df.tail()


# In[12]:


df.drop(['neg','neu','pos','compound'], axis=1, inplace=True)
df.tail()


# In[13]:


df.set_index('Date', inplace=True, drop=True)
df.tail()


# In[14]:


df.index = pd.DatetimeIndex(df.index)


# In[15]:


df.tail()


# In[16]:


df = df[~df.index.duplicated(keep='first')]


# In[23]:


idx = pd.date_range('2000-08-23','2021-04-22')

df = df.reindex(idx, fill_value=0)

print(df)

import matplotlib.pyplot as plt
plt.plot(df.index,df["polarity"])

plt.gcf().set_size_inches(20, 10)

plt.ylabel("Polarity", fontsize=18)
plt.xlabel("Date", fontsize=18)
plt.title("Sentiment of News Articles", fontsize=18)

plt.show()


# In[24]:


df.to_csv('sa_news.csv')


# In[18]:


# create data for Pie Chart
pichart = count_values_in_column(sentiment,"sentiment")
names= pichart.index
size=pichart["Percentage"]

#Creating PieCart
labels = ['Positive ['+str(size[1])+'%]' , 'Neutral ['+str(0)+'%]','Negative ['+str(size[0])+'%]']
sizes = [positive, negative]
colors = ['green', 'lightblue','red']
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title("Sentiment Analysis Result for keyword= "+keyword+"" )
plt.axis('equal')
plt.show()


# In[ ]:




