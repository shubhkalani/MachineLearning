import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

Datatrain = pd.read_csv("train.csv")
Datatest = pd.read_csv("test.csv")

cols_to_use = ['name','desc']
len_feats = ['name_len','desc_len']
count_feats = ['name_count','desc_count']

#for i in np.arange(2):
#   Datatest[len_feats[i]] = Datatest[cols_to_use[i]].apply(str).apply(len)
#  Datatest[count_feats[i]] = Datatest[cols_to_use[i]].apply(str).apply(lambda x: len(x.split(' ')))

#for i in np.arange(2):
#   Datatrain[len_feats[i]] = Datatrain[cols_to_use[i]].apply(str).apply(len)
#    Datatrain[count_feats[i]] = Datatrain[cols_to_use[i]].apply(str).apply(lambda x: len(x.split(' ')))




#Dividing into dependent and independent Data sets
x_train = Datatrain.iloc[:,[3,8,9,10,11]]
y_train = Datatrain.iloc[:,13]

x_test = Datatest.iloc[:,[3,8,9,10,11]]

from sklearn.preprocessing import StandardScaler
sc_x  = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators = 200,random_state = 0)
regressor.fit(x_train,y_train)

y_pred =regressor.predict(x_test)





nBsub = pd.DataFrame({'project_id':Datatest['project_id'],'final_status':y_pred})
nBsub = nBsub[['project_id','final_status']]
nBsub.to_csv("nBstarterwithoutLengthcolumns2.csv",index = False)










