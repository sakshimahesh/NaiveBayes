#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
data=pd.read_csv(r"C:\Users\saksh\Downloads\echocardiogram-modified.csv")# echocardiogram-modified.csv
print(data)
a=data[['A','B','C','D','E','F']]
b=data['Class']
atrain, atest, btrain, btest = train_test_split(a, b, train_size=0.66)
print(len(atrain))
print(len(btrain))
data.isna().any()
m = GaussianNB() #modeltype
#x=
m.fit(atrain,btrain)
m.score(atest,btest)
#btest();
x=m.predict(atest) 
print(accuracy_score(btest,x))
#m.predict(atest)))

