#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import tifffile


# In[7]:


#create a list with the 3D normalised vectors for each all stacks in the training dataset
total = []

point_1 = np.array([166,210,100])
point_2 = np.array([284,297,129])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([189,203,103])
point_2 = np.array([287,296,130])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([178,211,95])
point_2 = np.array([289,297,128])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([171,215,98])
point_2 = np.array([292,298,130])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([180,163,90])
point_2 = np.array([297,285,124])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([177,184,96])
point_2 = np.array([309,285,124])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([189,164,86])
point_2 = np.array([294,305,128])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([170,194,95])
point_2 = np.array([300,270,124])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([181,192,97])
point_2 = np.array([298,299,123])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([177,187,88])
point_2 = np.array([282,295,129])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([172,181,89])
point_2 = np.array([275,285,130])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([175,182,85])
point_2 = np.array([285,279,126])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([161,204,96])
point_2 = np.array([281,289,128])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([169,197,96])
point_2 = np.array([286,299,129])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([164,243,105])
point_2 = np.array([301,291,123])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([164,257,114])
point_2 = np.array([290,293,130])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([155,248,111])
point_2 = np.array([290,296,127])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([167,254,105])
point_2 = np.array([282,298,125])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([170,230,113])
point_2 = np.array([295,280,126])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([168,242,109])
point_2 = np.array([295,282,119])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([196,211,108])
point_2 = np.array([326,264,113])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([166,244,113])
point_2 = np.array([288,288,130])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([165,251,109])
point_2 = np.array([301,291,128])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([184,217,101])
point_2 = np.array([289,312,126])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

total = np.asarray(total)
for_validation = total


# In[8]:


#Adapted from https://blog.floydhub.com/introduction-to-k-means-clustering-in-python-with-scikit-learn/
    
get_ipython().run_line_magic('matplotlib', '')
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

#interpolate for direct comparison with other methods
total =  np.interp(total, (total.min(), total.max()), (-1, +1))

#apply k-means algorithm with 2 clusters on total
kmeans = KMeans(n_clusters=2, random_state=0).fit(total)

# Get the cluster centroids
print('centroids: ', kmeans.cluster_centers_)
# Get the cluster labels
print(kmeans.labels_)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#mutant and healthy hearts
mutants = total[0:14, :]
print('Mutants: ', mutants)
healthy = total[14:23, :]
print('Healthy: ', healthy)

#set axis parameters and plot the data points together with the k-means clusters centroids
ax.tick_params(axis='both', which='major', labelsize=25)
ax.tick_params(axis='both', which='minor', labelsize=25)
ax.scatter(healthy[:,0], healthy[:,1], healthy[:,2], marker='o', c='blue', s=300, label='Healthy hearts')
ax.scatter(mutants[:,0], mutants[:,1], mutants[:,2], marker='o', c='orange', s=300, label='Mutant hearts')
ax.scatter(total[23,0], total[23,1], total[23,2], marker='o', c='orange', edgecolors='red', linewidth=4, s=300, label='Misclassified healthy hearts')
ax.scatter(kmeans.cluster_centers_[0,0],  kmeans.cluster_centers_[0,1], kmeans.cluster_centers_[0,2], c='black', marker='x', linewidth=5, s=450, label='Mutant centroid')
ax.scatter(kmeans.cluster_centers_[1,0],  kmeans.cluster_centers_[1,1], kmeans.cluster_centers_[1,2], c='green', marker='x', linewidth=5, s=450, label='Healthy centroid')
ax.grid(False)

#calculate homogeneity, completeness and V-score
labels_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
labels_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
print('Homogeneity = ', metrics.homogeneity_score(labels_true, labels_pred))
print('Completeness = ', metrics.completeness_score(labels_true, labels_pred))
print('v-score = ', metrics.v_measure_score(labels_true, labels_pred))


# In[11]:


validation = []

point_1 = np.array([204,198,151])
point_2 = np.array([273,325,212])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
validation.append(vec_norm)

point_1 = np.array([129,172,94])
point_2 = np.array([245,282,119])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
validation.append(vec_norm)

point_1 = np.array([141,195,116])
point_2 = np.array([249,331,158])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
validation.append(vec_norm)

point_1 = np.array([150,226,105])
point_2 = np.array([293,298,130])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
validation.append(vec_norm)

point_1 = np.array([144,182,111])
point_2 = np.array([297,245,131])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
validation.append(vec_norm)

point_1 = np.array([161,215,77])
point_2 = np.array([281,266,134])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
validation.append(vec_norm)
print(validation)

validation = np.asarray(validation)
validation = np.interp(validation, (for_validation.min(), for_validation.max()), (-1, +1))

print('interpolated: ', validation)
plt3d = plt.figure().gca(projection='3d')
ax = plt3d

ax.tick_params(axis='both', which='major', labelsize=25)
ax.tick_params(axis='both', which='minor', labelsize=25)
ax.scatter(validation[0:3,0], validation[0:3,1], validation[0:3,2], marker='o', c='orange', s=300, label='Mutant hearts')
ax.scatter(validation[3:6,0], validation[3:6,1], validation[3:6,2], marker='o', c='blue', s=300, label='Healthy hearts')
ax.scatter(kmeans.cluster_centers_[0,0],  kmeans.cluster_centers_[0,1], kmeans.cluster_centers_[0,2], c='black', marker='x', linewidth=5, s=450, label='Mutant centroid')
ax.scatter(kmeans.cluster_centers_[1,0],  kmeans.cluster_centers_[1,1], kmeans.cluster_centers_[1,2], c='green', marker='x', linewidth=5, s=450, label='Healthy centroid')
ax.grid(False)

plt.show()


# In[ ]:




