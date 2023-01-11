

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
import string
from gensim.parsing.preprocessing import remove_stopwords




reviews_train = []
labels_train = []

reviews_test = []
labels_test = []

"""Train_negative"""

for i in range(1,4):
  file_review = open(f'/content/drive/MyDrive/Data/Reviews_Data/train/negative/{i}.review','r')
  check = "False"
  for file in file_review:
   if file == '</review_text>\n':
    check = "False"
   if check =="True":
     if file != "\n":
        reviews_train.append(file)
        labels_train.append("negative")
   if file == '<review_text>\n':
    check = "True"

"""Train_positive"""

for i in range(1,4):
  file_review = open(f'/content/drive/MyDrive/Data/Reviews_Data/train/positive/{i}.review','r')
  check = "False"
  for file in file_review:
   if file == '</review_text>\n':
    check = "False"
   if check =="True":
     if file != "\n":
        reviews_train.append(file)
        labels_train.append("positive")
   if file == '<review_text>\n':
    check = "True"

"""Test_negative"""

file_review = open(f'/content/drive/MyDrive/Data/Reviews_Data/test/negative/negative.review','r')
  check = "False"
  for file in file_review:
   if file == '</review_text>\n':
    check = "False"
   if check =="True":
     if file != "\n":
        reviews_test.append(file)
        labels_test.append("negative")
   if file == '<review_text>\n':
    check = "True"

"""Test_positive"""

file_review = open(f'/content/drive/MyDrive/Data/Reviews_Data/test/positive/positive.review','r')
  check = "False"
  for file in file_review:
   if file == '</review_text>\n':
    check = "False"
   if check =="True":
     if file != "\n":
        reviews_test.append(file)
        labels_test.append("positive")
   if file == '<review_text>\n':
    check = "True"

"""Preprocessing"""

def clean_text(text):
  text = text.lower()
  text = remove_stopwords(text)
  text = re.sub('\[.*?\]','',text)
  text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
  text = re.sub('\w*\d\w*','',text)
  text = re.sub('\n','',text)
  return text

# Train
for i in range(0,len(reviews_train)):
      reviews_train[i] = clean_text(reviews_train[i])

# Test
for i in range(0,len(reviews_test)):
      reviews_test[i] = clean_text(reviews_test[i])

"""Data"""

reviews_train = np.array(reviews_train).reshape(-1,1)
labels_train = np.array(labels_train).reshape(-1,1)
Data_train = np.concatenate([reviews_train,labels_train],axis=1)
Data_train = pd.DataFrame(Data_train,columns=['review','label'])

reviews_test = np.array(reviews_test).reshape(-1,1)
labels_test = np.array(labels_test).reshape(-1,1)
Data_test = np.concatenate([reviews_test,labels_test],axis=1)
Data_test = pd.DataFrame(Data_test,columns=['review','label'])

"""Model"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

t = TfidfVectorizer()
c = LogisticRegression(solver='lbfgs')
model = Pipeline([('vectorizer',t),('classifier',c)])
model.fit(Data_train['review'], Data_train['label'])

"""Test"""

from sklearn.metrics import accuracy_score

print(np.round(accuracy_score(model.predict(Data_test['review']),Data_test['label']),2))