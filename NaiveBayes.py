import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv




dataset = pd.read_csv('pima-indians-diabetes.data',header=None)
dataset.columns = ['pregnant', 'plasma-glucose-conc','blood pressure','skin thick','insulin','bmi','pedigree function','age','class']





#Breaking into dependent and independent datasets
x = dataset.iloc[:,:8].values
y = dataset.iloc[:,8].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=0,strategy='mean',axis = 0)
imputer = imputer.fit(x[:,1:])
x[:,1:]=imputer.transform(x[:,1:])



#splitting dataset into test and train dataset
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)



#USING multinomial NB before standard scaling,as it does not support negative values
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)
#predicting results
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm1 = confusion_matrix(y_test,y_pred)

s1 = accuracy_score(y_test,y_pred)
print "Accuracy of Multinomial NB is",s1






#Standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#using Gaussian naive bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

#predicting results
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

s = accuracy_score(y_test,y_pred)
print "Accuracy of Gaussian NB is",s

ma