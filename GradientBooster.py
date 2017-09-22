#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:57:14 2017

@author: shubh
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection

Datatrain = pd.read_csv("train.csv")
Datatest = pd.read_csv("test.csv")

cols_to_use = ['name','desc']
len_feats = ['name_len','desc_len']
count_feats = ['name_count','desc_count']

for i in np.arange(2):
    Datatest[len_feats[i]] = Datatest[cols_to_use[i]].apply(str).apply(len)
    Datatest[count_feats[i]] = Datatest[cols_to_use[i]].apply(str).apply(lambda x: len(x.split(' ')))

for i in np.arange(2):
    Datatrain[len_feats[i]] = Datatrain[cols_to_use[i]].apply(str).apply(len)
    Datatrain[count_feats[i]] = Datatrain[cols_to_use[i]].apply(str).apply(lambda x: len(x.split(' ')))




#Dividing into dependent and independent Data sets
x_train = Datatrain.iloc[:,[3,8,9,10,11,14,15,16,17]]
y_train = Datatrain.iloc[:,13]

x_test = Datatest.iloc[:,[3,8,9,10,11,12,13,14,15]]

from sklearn.preprocessing import StandardScaler
sc_x  = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

model = GradientBoostingClassifier(n_estimators=300,learning_rate=1, random_state=10,warm_start=True)
model.fit(x_train,y_train)


#y_pred = np.array(y_pred,dtype=int)
y_pred = model.predict(x_test)

#y_pred = np.array(y_pred,dtype=int)




nBsub = pd.DataFrame({'project_id':Datatest['project_id'],'final_status':y_pred})
nBsub = nBsub[['project_id','final_status']]
nBsub.to_csv("nBstarterGradBoost.csv",index = False)





