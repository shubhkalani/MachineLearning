import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the csv data file
iq = pd.read_csv("iq_size.csv")

#Breaking the data into dependent and independent sets
x = iq.iloc[:,1:4].values
y = iq.iloc[:,0].values

           
#Importing and applying Linear Regression           
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

#predicting the results for given data
y_pred = regressor.predict((90,70,150))
print y_pred

#Building the optimal model 
import statsmodels.formula.api as sm
x = np.append(arr=np.ones((38,1)).astype(int),values = x, axis = 1)
x_opt = x[:,[0,1,2,3]]
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
print regressor_ols.summary()

#Removing the column with highest p

x_opt = x[:,[0,1,2]]
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
print regressor_ols.summary()

print "According to regressor summary() Brain and Height are more useful in predicting than the weight variable"
