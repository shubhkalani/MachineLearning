import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Dataset = pd.read_csv("bluegills.csv")


#Breaking the data into Dependent variable and independent variable.

x = Dataset.iloc[:,0:1].values
y = Dataset.iloc[:,1].values


#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#fitting polynomial Regresssion 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
print poly_reg.fit(x_poly,y)
lin_reg2 =  LinearRegression()
lin_reg2.fit(x_poly,y)


#Visualising the Linear Regression
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color = 'blue')
plt.title("Fish length v/s its age(Linear regression)")
plt.xlabel('Age')
plt.ylabel('Lenght')
plt.show()


#Visualising the polynomial Regression model
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color = 'blue')
plt.title("Fish length v/s its age(Linear regression)")
plt.xlabel('Age')
plt.ylabel('Length')
plt.show()


print "According to visual patterns,Scattering in polynomial is less than linear,so polynomial model is more correct"

#Predicting the Bluegil fish length of age 5
print "Age of 5 year old fish is using polynomial model"
print lin_reg2.predict(poly_reg.fit_transform(5))

