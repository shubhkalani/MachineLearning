import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Dataset = pd.read_csv("mushrooms.csv")
x = Dataset.iloc[:,1:23].values
y = Dataset.iloc[:,0:1].values

#Performing Labelencoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x = LabelEncoder()
labelencoder_y = LabelEncoder()

for i in range(22):
        x[:,i] = labelencoder_x.fit_transform(x[:,i])

y = labelencoder_y.fit_transform(y)
        

        
x=pd.DataFrame(x)
y=pd.DataFrame(y)


#Onehotencoding on habitant,population and odor

onehotencoder = OneHotEncoder(categorical_features = [4,20,21])
x = onehotencoder.fit_transform(x).toarray()






#Fitting k-NN to taining set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric = 'minkowski',p=2)
classifier.fit(x,y)

y_pred = classifier.predict(x)
#MakingConfusionMatrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,y_pred)

print classifier.score(x,y)


