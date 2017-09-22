import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Dataset = pd.read_csv("tshirts.csv")

x=Dataset.iloc[:,[1,2]].values

#Clustering from kmeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3,init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans ==0,0],x[y_kmeans ==0,1],s=100,c='red',label = 'Cluster 1')
plt.scatter(x[y_kmeans ==1,0],x[y_kmeans ==1,1],s=100,c='blue',label = 'Cluster 2')
plt.scatter(x[y_kmeans ==2,0],x[y_kmeans ==2,1],s=100,c='green',label = 'Cluster 3')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title("cluster of tshirts")
plt.xlabel('height')
plt.ylabel('weight')
plt.legend()
plt.show()

print """accordingly we can see that cluster 3 is small,cluster1 is medium,cluster2 is large size """
