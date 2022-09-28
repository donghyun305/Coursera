# Force no warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# Setup and imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle

plt.rcParams['figure.figsize'] = [6,6]
sns.set_style('whitegrid')
sns.set_context('talk')

def display_cluster(X, km=[], num_clusters=0):
    color = 'brgcmyk'
    alpha = 0.5
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0], X[:,1], c = color[0], alpha=alpha, s=s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[km.labels_ == i,0], X[km.labels_ == i, 1], c = color[i], alpha=alpha, s=s)
            plt.scatter(km.cluster_centers_[i][0], km.cluster_centers_[i][1], c = color[i], marker = 'x', s=100)

angle = np.linspace(0, 2*np.pi, 20, endpoint=False)
X = np.append([np.cos(angle)], [np.sin(angle)], 0).transpose()
display_cluster(X)

num_clusters = 2
km = KMeans(n_clusters=num_clusters, random_state=10, n_init=1)
km.fit(X)
display_cluster(X, km, num_clusters)
# plt.show()

# Determining Optimium number of Clusters
n_samples = 1000
n_bins = 4
centers = [(-3, -3), (0,0), (3,3), (6,6)]
X, y = make_blobs(n_samples = n_samples, n_features=2, cluster_std=1.0,
                  centers = centers, shuffle=False, random_state=42)
display_cluster(X)
# plt.show()

# run with 7 clusters
num_clusters = 7
km = KMeans(n_clusters=num_clusters)
km.fit(X)
display_cluster(X, km, num_clusters)
# plt.show()

inertia = []
list_num_clusters = list(range(1,11))
for num_clusters in list_num_clusters:
    km = KMeans(n_clusters=num_clusters)
    km.fit(X)
    inertia.append(km.inertia_)
plt.plot(list_num_clusters, inertia)
plt.scatter(list_num_clusters, inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
# plt.show()

#Clustering Colors from an Image
img = plt.imread('peppers.jpg', format='jpeg')
plt.imshow(img)
plt.axis('off')

R = 35
G = 95
B = 131
plt.imshow([[np.array([R,G,B]).astype('uint8')]])
plt.axis('off')
plt.show()