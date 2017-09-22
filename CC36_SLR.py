import pandas as pd
import matplotlib.pyplot as plt
Foodtruck = pd.read_csv("Foodtruck.csv")

#Splitting into dependent and independent columns
X=Foodtruck.iloc[:,0:1]
Y=Foodtruck.iloc[:,1]

#Splitting into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)


#Generating linera regression prediction model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


#Predicting the Test set Results
y_pred = regressor.predict(X_test)

#Visusalising the Training set Results
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Profit vs Population(Training set)")
plt.xlabel("population")
plt.ylabel("Profit")
plt.show()

#visualising the test set result
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Profit vs Population(Test set)")
plt.xlabel("population")
plt.ylabel("Profit")
plt.show()

#Estimated profit when population is 3.073million
populationjp = 3.073
Profit_jaipur = regressor.predict(populationjp)
print "profit of jaipur is",Profit_jaipur,"million"


