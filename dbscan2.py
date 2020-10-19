import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib.pyplot as plt

#data = np.loadtxt(r"/users/mutukula/Desktop/netmate_mixed_before_injection.out", delimiter=',', usecols=[9,10,11,12,13,14,15,16])
#data = np.loadtxt(r"/users/mutukula/Desktop/netmate_mixed_before_injection.out", delimiter=',', usecols=[9,10,11,12,13,14,15,16,44,45,46,47,48,49,50,51,52,53,54])
data = np.loadtxt(r"/users/mutukula/Desktop/netmate_mixed.out", delimiter=',', usecols=[9,10,11,12,13,14,15,16])
#data = np.loadtxt(r"/users/mutukula/Desktop/netmate_mixed.out", delimiter=',', usecols=[9,10,11,12,13,14,15,16,44,45,46,47,48,49,50,51,52,53,54])
#data = np.loadtxt(r"/users/mutukula/Desktop/netmate_only_malware.out",delimiter=',', usecols=[9,10,11,12,13,14,15,16])
#data = np.loadtxt(r"/users/mutukula/Desktop/netmate_only_malware.out", delimiter=',', usecols=[9,10,11,12,13,14,15,16,44,45,46,47,48,49,50,51,52,53,54])



#data = pd.read_excel("netmate_mixed.xlsx",skiprows=[0])
#data = pd.read_excel("netmate_mixed_no_duplicates_without.xlsx",skiprows=[0])
#data = pd.read_excel("netmate_mixed_before_injection_no_duplicates.xlsx",skiprows=[0])
#data = pd.read_excel("netmate_mixed_before_injection_without_no_duplicates.xlsx",skiprows=[0])







X = StandardScaler().fit_transform(data)
X = MinMaxScaler().fit_transform(X)

"""
neigh = NearestNeighbors(n_neighbors=100)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
print("Printing distances here")
print(distances)
print("Printing first row here")
print(distances[0])


distances = np.mean(distances, axis=1)
print("After finding mean")
print(distances)
print("First row")
print(distances[0])
distances = np.sort(distances, axis=0)
print("After sorting")
print(distances)
print("First row")
print(distances[0])
print("Last row")
print(distances[-1])

plt.plot(distances)
plt.show()

"""

# Compute DBSCAN
#db = DBSCAN(eps=0.11, min_samples=100).fit(X) #without
db = DBSCAN(eps=0.13, min_samples=100).fit(X) #with




core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print(labels)


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)  
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

