import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Dataset = pd.read_csv("affairs.csv")
x = Dataset.iloc[:,0:8].values
y = Dataset.iloc[:,-1].values
                

#Encoding categorical data occupation and occupation_husb
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [6,7])
x = onehotencoder.fit_transform(x).toarray()

#Avoiding dummy variable trap
x1 = x[:,1:6]
x2= x[:,7:]
x = np.append(x1,x2,axis = 1)


#Breaking into test and train dataset
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)







#fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)



#Using .score function
print "accuracy is"
print classifier.score(x_train,y_train)



#percntage of total women actually having an affair
print "percentage of women actually having an affair"
print Dataset["affair"].value_counts(normalize = True)[1]*100

             
#Building optimal Model

#Addition of intercept
import statsmodels.formula.api as sm
x = np.append(arr=np.ones((6366,1)).astype(int),values = x,axis = 1)

regressor_OLS=sm.OLS(endog=y,exog=x).fit()
print regressor_OLS.summary()
x=pd.DataFrame(x)
x=x.drop([16],axis=1)

regressor_OLS=sm.OLS(endog=y,exog=x).fit()
print regressor_OLS.summary()

x=x.drop([14],axis=1)

regressor_OLS=sm.OLS(endog=y,exog=x).fit()
print regressor_OLS.summary()

x=x.drop([1],axis=1)

regressor_OLS=sm.OLS(endog=y,exog=x).fit()
print regressor_OLS.summary()

x=x.drop([10],axis=1)

regressor_OLS=sm.OLS(endog=y,exog=x).fit()
print regressor_OLS.summary()

x=x.drop([8],axis=1)

regressor_OLS=sm.OLS(endog=y,exog=x).fit()
print regressor_OLS.summary()

x=x.drop([9],axis=1)

regressor_OLS=sm.OLS(endog=y,exog=x).fit()
print regressor_OLS.summary()

x=x.drop([6],axis=1)

regressor_OLS=sm.OLS(endog=y,exog=x).fit()
print regressor_OLS.summary()

x=x.drop([3],axis=1)

regressor_OLS=sm.OLS(endog=y,exog=x).fit()
print regressor_OLS.summary()

x=x.drop([7],axis=1)

regressor_OLS=sm.OLS(endog=y,exog=x).fit()
print regressor_OLS.summary()


print "efficient variables are"
print "rate_mariage,age,yrs_married,childern,occupation lady"



#"""Predicting result for 25year old woman,teacher=4,college_graduate=16,married=3years,child=1,religious=4,marriage=3,occup_husb=2)
#Dataset for woman is (3,25,3,1,4,16,4,2)
p = [3,25,3,1,4,16,4,2]
#Encoding categorical data occupation and occupation_husb
px=[0,0,0,1,0,0,1,0,0,0,3,25,3,1,4,16]


print "probability of having affair of lady is"
print classifier.predict_proba(px)[0][0]




