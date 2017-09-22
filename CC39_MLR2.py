import pandas as pd
import numpy as np

Dataset = pd.read_csv("stats_females.csv")
#Defining Dependent and Independent Components
x = Dataset.iloc[:,1:].values
y = Dataset.iloc[:,0].values
                
#Importing and applying Linear Regression           
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

#Building the optimal model (1)
import statsmodels.formula.api as sm
x = np.append(arr=np.ones((214,1)).astype(int),values = x, axis = 1)
x_opt = x[:,[0,1,2]]
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
print regressor_ols.summary()
print "so According to above study both the parents height are of same importance."

#When Father'sheight isConstant,we will take mother's Coefficient.
print "When father's height is constant for every inch increase in mother's height daughters height is increasing by .3035 inch"
#When Mother's height isConstant,we will take Father's Coefficient.
print "When Mother's's height is constant for every inch increase in father's height daughters height is increasing by .3879 inch"

 




