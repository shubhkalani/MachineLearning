import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
iris = load_iris()
iris = iris.data

#Breaking the dataset into dependent and independent variables:

#Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
iris=pca.fit_transform(iris)
explained_variance = pca.explained_variance_ratio_

#Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3,init = 'k-means++',random_state = 0)
y_kmeans = kmeans.fit_predict(iris)

#Visualisation
plt.scatter(iris[y_kmeans == 0,0],iris[y_kmeans == 0,1],s=100,c='red',label = 'cluster 1')
plt.scatter(iris[y_kmeans == 1,0],iris[y_kmeans == 1,1],s=100,c='green',label = 'cluster 2')
plt.scatter(iris[y_kmeans == 2,0],iris[y_kmeans == 2,1],s=100,c='blue',label = 'cluster 3')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='yellow',label='centroids')
plt.title('cluster of various iris categories')
plt.xlabel('iris')
plt.ylabel('various legthns and breadths of petals')
plt.legend()
plt.show()
