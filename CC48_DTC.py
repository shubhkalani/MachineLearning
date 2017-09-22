import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("tree_addhealth.csv")
#Handling Nan Values
dataset = dataset.fillna(dataset.mean())
              
#Breaking into Dependent and independent variables:
x=dataset.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15]].values
y=dataset.iloc[:,7:8].values
              

#Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x = sc.fit_transform(x)


#Building a decision tree classification model:
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
classifier.fit(x,y)

#Prediction
y_pred = classifier.predict(x)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,y_pred)

#Visualising the results
from matplotlib.colors import ListedColormap
x_set,y_set = x,y
x1,x2 =np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop =x_set[:,0].max()+1,step =0.01),np.arange(start =x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):plt.scatter(x_set[y_set ==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label = j)
plt.title('Decision tree classification')
plt.xlabel('Various Attributes')
plt.ylabel('Smoke')
plt.legend()
plt.show()