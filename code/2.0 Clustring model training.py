#
# Program: 2.0 Clustering model training
#
# Purpose: Train k-means and K-modes
#
# Written by: Qiuhua Liu(11258799) 
#             Yanhan Peng(11125583) 
#             Chaoyang Zheng(11249259) 
# 
# Updated: Dec 2019
#        
# ------------------------------------------------------.



# coding: utf-8

# ###### Import required packages

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os


# ###### loading data 

data = os.path.abspath(r"C:/Users/Zheng Chaoyang/Desktop/ML Group project/Data/Output")
os.chdir(data)
train = pd.read_csv("intem_nodup.csv")
train.head()


# ##### One hot encoding to transfer cat to binary

list_catvar = ["item_nbr","store_nbr", "item_class","store_city","store_state","store_type","store_cluster"]
for i in list_catvar:
  train[i] = pd.Categorical(train[i])
  dfDummies = pd.get_dummies(train[i], prefix = i)
  train = pd.concat([train, dfDummies], axis=1)
train.head()
train.info()

###################
## Train K-means ##
###################

# Selecting columns as x 
list_col = list(train.columns) 
## iterating the columns 
Selec_dummy_x = []
for i in list_col:
  for j in list_catvar:
    if  j in i:
      Selec_dummy_x.append(i)
  
# If we will include the original parameters

for i in list_catvar:
  Selec_dummy_x.remove(i)


# Selecting variables (continuous and dummy) to train the modedl 
Selected_x = ["item_perishable"]
Selected_x = Selected_x + Selec_dummy_x
X_kmeans  = np.asarray(train[Selected_x[0:len(Selected_x)]])

# Train model
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k, random_state=0)
    km = km.fit(X_kmeans)
    Sum_of_squared_distances.append(km.inertia_)

# Plot 
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

##################
## Train Kmodes ##
##################
K = range(1,15)
for k in K:
    km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=1)
    clusters = km.fit_predict(data)

    
    
km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(data)

# Print the cluster centroids
print(km.cluster_centroids_)

# Train model
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k, random_state=0)
    km = km.fit(X_kmeans)
    Sum_of_squared_distances.append(km.inertia_)

for n_clusters in range_n_clusters:
    clusterer = KModes(n_clusters=n_clusters)
    centers = clusterer.cluster_centers_

    import numpy as np
from kmodes.kmodes import KModes

# random categorical data
data = np.random.choice(20, (100, 10))

km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(data)


