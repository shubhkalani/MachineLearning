import pandas as pd
import matplotlib.pyplot as plt
Movie = pd.read_csv("Bahubali2_vs_Dangal.csv")

#Splitting into dependent and independent columns
Days=Movie.iloc[:,0:1]
Bahubali2=Movie.iloc[:,1]
Dangal=Movie.iloc[:,2]



#Generating linera regression prediction model for Bahubli2
from sklearn.linear_model import LinearRegression
regressorBahu = LinearRegression()
regressorBahu.fit(Days, Bahubali2)


#Generating linera regression prediction model for Dangal
from sklearn.linear_model import LinearRegression
regressorDang = LinearRegression()
regressorDang.fit(Days, Dangal)

#Visusalising the Results of dangal and Bahubali2 in graph
plt.scatter(Days,Bahubali2,color='red')
plt.plot(Days,regressorBahu.predict(Days),color='red')
plt.scatter(Days,Dangal,color='blue')
plt.plot(Days,regressorDang.predict(Days),color='blue')
plt.title("Earning of Dangal and bahubali Day wise collection.(Red Denotes Bahubali,Blue Denotes Dangal)")
plt.xlabel("Day")
plt.ylabel("Collection")
plt.show()


#Predicting the Results for 10th day
RegressorBahu = regressorBahu.predict(10)

RegressorDang = regressorDang.predict(10)


if RegressorBahu>RegressorDang:
    print "Bahubali2 earned more on 10th day"
else:
    print "Dangal earned more on 10th day"

