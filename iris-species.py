import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the iris dataset
irisdataset = pd.read_csv("Iris.csv")
irisdataset

# Drop the Id and Species columns
colunsdrop = ['Id', 'Species']
irisdatasetdrop = irisdataset.drop(colunsdrop, axis=1)
irisdatasetdrop

# Using KMenans
kmeans = KMeans(n_clusters=3, random_state=49, n_init='auto')
clustering_kmeans = kmeans.fit_predict(irisdatasetdrop)
irisdatasetdrop['Cluster K-Means'] = clustering_kmeans
irisdatasetdrop

# Plot the clusters
labels_kmeans1 = irisdatasetdrop[clustering_kmeans == 1]
labels_kmeans2 = irisdatasetdrop[clustering_kmeans == 2]
labels_kmeans0 = irisdatasetdrop[clustering_kmeans == 0]
plt.scatter(irisdatasetdrop['PetalLengthCm'].where(irisdatasetdrop['Cluster K-Means']==1), irisdatasetdrop['SepalWidthCm'].where(irisdatasetdrop['Cluster K-Means']==1), color='black')
plt.scatter(irisdatasetdrop['PetalLengthCm'].where(irisdatasetdrop['Cluster K-Means']==2), irisdatasetdrop['SepalWidthCm'].where(irisdatasetdrop['Cluster K-Means']==2), color='blue')
plt.scatter(irisdatasetdrop['PetalLengthCm'].where(irisdatasetdrop['Cluster K-Means']==0), irisdatasetdrop['SepalWidthCm'].where(irisdatasetdrop['Cluster K-Means']==0), color='red')
plt.show()


New_Species = {
    'Iris-setosa': '0',
    'Iris-versicolor': '1',
    'Iris-virginica': '-1'
}
irisdataset_new = irisdataset['Species'].replace(New_Species)
irisdataset['New_Species'] = irisdataset_new
irisdataset

# Using DB-Scan
clustering = DBSCAN(eps=0.9, min_samples=42).fit(irisdatasetdrop)
clustering.labels_
irisdatasetdrop['Cluster DB-Scan'] = clustering.labels_
irisdatasetdrop

# Plot the clusters
labels_dbscan0 = irisdatasetdrop[clustering.labels_ == 0]
labels_dbscan1 = irisdatasetdrop[clustering.labels_ == 1]
labels_dbscan2 = irisdatasetdrop[clustering.labels_ == -1]

plt.scatter(irisdatasetdrop['PetalLengthCm'].where(irisdatasetdrop['Cluster DB-Scan']==0), irisdatasetdrop['PetalWidthCm'].where(irisdatasetdrop['Cluster DB-Scan']==0), color='black')
plt.scatter(irisdatasetdrop['PetalLengthCm'].where(irisdatasetdrop['Cluster DB-Scan']==1), irisdatasetdrop['PetalWidthCm'].where(irisdatasetdrop['Cluster DB-Scan']==1), color='blue')
plt.scatter(irisdatasetdrop['PetalLengthCm'].where(irisdatasetdrop['Cluster DB-Scan']==-1), irisdatasetdrop['PetalWidthCm'].where(irisdatasetdrop['Cluster DB-Scan']==-1), color='red')