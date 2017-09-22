import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Dataset = pd.read_csv("deliveryfleet.csv")

x=Dataset.iloc[:,[1,2]].values
             

#Using elbow method to find the optimal numbers of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters = i,init = 'k-means++', random_state = 0)
    Kmeans.fit(x)
    wcss.append(Kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow MEthod')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

print "Optimal model is 3 according to Elbow model"



kmeans = KMeans(n_clusters = 2,init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#plotting resuls
plt.scatter(x[y_kmeans ==0,0],x[y_kmeans ==0,1],s=100,c='red',label = 'Cluster 1')
plt.scatter(x[y_kmeans ==1,0],x[y_kmeans ==1,1],s=100,c='blue',label = 'Cluster 2')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title("cluster of urban and rural")
plt.xlabel('Distance feature')
plt.ylabel('speeding feature')
plt.legend()
plt.show()


#Defining speeding vs urban rural
print """ Here in cluster 1,our speeding limit is crossed few times,so accordingly it is urban,
          whereas in cluster 2,our speeding limit is crossed much more than cluster 1,so it is rural."""


