import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Dataset = pd.read_csv("PastHires.csv")

#splitting into dependent and independent
x = Dataset.iloc[:,[0,1,2,3,4,5]].values
y = Dataset.iloc[:,6].values
                


#LabelEncoding of data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x  = LabelEncoder()


for i in [1,3,4,5]:
    print i
    x[:,i]=labelencoder_x.fit_transform(x[:,i])

#onehotencoder = OneHotEncoder(categorical_features=[1,3,4,5])
#x = onehotencoder.fit_transform(x).toarray()


x=pd.DataFrame(x)
    
y=labelencoder_x.fit_transform(y)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)

y_pred = regressor.predict(x)

#Confussion Metrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,y_pred)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,random_state=0)
classifier.fit(x,y)

y_pred1 = classifier.predict(x)

cm1=confusion_matrix(y,y_pred1)