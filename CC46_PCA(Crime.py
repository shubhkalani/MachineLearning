import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Dataset = pd.read_csv("crime_data.csv")

#Splitting into Dependent and Independent Data
x=Dataset.iloc[:,[2,3,4]].values  #Removed the urban pop as it is not taken for clustering
y=Dataset.iloc[:,0:1].values

#labelEncoding of y:
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

#Using K-means for clustering (3 Clusters)
from sklearn.cluster import KMeans
kmeans  = KMeans(n_clusters=3,init = 'k-means++',random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#Visualising the clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans ==0,1],s=100,c='red',label = 'cluster1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans ==1,1],s=100,c='green',label = 'cluster2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans ==2,1],s=100,c='blue',label = 'cluster3')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title("clusters of Cities based on crimes")
plt.xlabel("Crime rate")
plt.ylabel("Cities")
plt.legend()
plt.show()
