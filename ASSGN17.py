from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
%matplotlib inline
from sklearn.svm import SVC
from sklearn.datasets import load_digits
digit=load_digits()
x=digit.data
y=digit.target
plt.imshow(x[4].reshape(8,8),cmap=plt.cm.gray)
from sklearn.cluster import KMeans
kmeans = KMeans()
kmeans
kmeans = KMeans(n_clusters=10,max_iter=2000,random_state=1)
predict=kmeans.fit_predict(x)
zero = np.zeros_like(predict)
from scipy.stats import mode
for a in range(10):
  mask = (predict==a)
  zero[mask] = mode(y[mask])[0]
from sklearn import metrics
metrics.accuracy_score(zero,y)
confusion_matrix(zero,y)
from sklearn.datasets import load_sample_image
china=load_sample_image('china.jpg')
plt.imshow(china)
china.shape
x, y, z = china.shape
china_2d = china.reshape(x*y, z)
china_2d.shape
kmeans_cluster = KMeans(n_clusters=7)
kmeans_cluster.fit(china_2d)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_centers #Seven different centroids
cluster_labels = kmeans_cluster.labels_
cluster_labels
china_2d
np.unique(cluster_labels)
array([0, 1, 2, 3, 4, 5, 6], dtype=int32)
picture = cluster_centers[cluster_labels]
picture
picture.shape
picture = picture.reshape(x,y,z)
picture.shape
plt.figure(figsize = (15,8))
plt.imshow(cluster_centers[cluster_labels].reshape(x,y,z).astype(np.uint8))







