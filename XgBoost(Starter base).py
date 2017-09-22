import pandas as pd
import numpy as np
import re
import datetime
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
#from nltk.stem.snowball import snowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb

#Loading train and test data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


#Convertinf into len_features
cols_to_use = ['name','desc']
len_feats = ['name_len','desc_len']
count_feats = ['name_count','desc_count']

for i in np.arange(2):
    test[len_feats[i]] = test[cols_to_use[i]].apply(str).apply(len)
    test[count_feats[i]] = test[cols_to_use[i]].apply(str).apply(lambda x: len(x.split(' ')))

for i in np.arange(2):
    train[len_feats[i]] = train[cols_to_use[i]].apply(str).apply(len)
    train[count_feats[i]] = train[cols_to_use[i]].apply(str).apply(lambda x: len(x.split(' ')))

