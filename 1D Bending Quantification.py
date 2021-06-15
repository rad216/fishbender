#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import tifffile


# In[16]:


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


# In[17]:


get_ipython().run_line_magic('matplotlib', '')
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import matplotlib.pyplot as plt
#print(total)
X = np.asarray([total[0][0], total[1][0], total[2][0], total[3][0], total[4][0], total[5][0], total[6][0], total[7][0], total[8][0], total[9][0], total[10][0], total[11][0], total[12][0], total[13][0], total[14][0], total[15][0], total[16][0], total[17][0], total[18][0], total[19][0], total[20][0], total[21][0], total[22][0], total[23][0]])
X=-X
for i in range(0,24):
   X[i] = str(round(X[i], 2))
   print('Stack ', i+1, ': ', X[i])

X = np.interp(X, (X.min(), X.max()), (-1, +1))
for i in range(0,24):
   X[i] = str(round(X[i], 2))
   print('Stack ', i+1, ': ', X[i])

X = np.asarray(X).reshape(-1, 1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#mutant and healthy hearts
print('after interpolation:', X)
mutants1 = X[0:7]
mutants2 = X[8:14]
mutants = np.vstack((mutants1, mutants2))
#print('Mutants: ', mutants)
healthy = X[14:23]
#print('Healthy: ', healthy)

ax.tick_params(axis='x', labelsize=50)

#ax.tick_params(axis='both', which='major', labelsize=30)
#ax.tick_params(axis='both', which='minor', labelsize=30)
# Get the cluster centroids
#print(kmeans.cluster_centers_)
#print(kmeans.cluster_centers_[0, 0])
#print(kmeans.cluster_centers_[1])
   
# Get the cluster labels
#print(kmeans.labels_)
# Plotting the cluster centers and the data points on a 2D plane
plt.scatter(mutants[:, 0], np.zeros_like(mutants) + 0, label='Mutant hearts', c='orange', marker='o', s=300)
plt.scatter(healthy[:, 0], np.zeros_like(healthy) + 0, label='Healthy hearts', c='blue', marker='o', s=300)
plt.scatter(X[7, 0], np.zeros(1) + 0, label='Misclassified mutant hearts', c='blue', edgecolors='red', linewidth=5, marker='o', s=300)
plt.scatter(X[23, 0], np.zeros(1) + 0, label='Misclassified healthy hearts', c='orange', edgecolors='red', linewidth=5, marker='o', s=300)
 
plt.scatter(kmeans.cluster_centers_[1], np.zeros_like(kmeans.cluster_centers_[1]) + 0, c='green', marker='x', linewidth=5, s=450, label='Healthy centroid')
plt.scatter(kmeans.cluster_centers_[0], np.zeros_like(kmeans.cluster_centers_[0]) + 0, c='black', marker='x', linewidth=5, s=450, label='Mutant centroid')


plt.vlines(0.32, -0.05, 0.05, linestyles='dashed', label = 'Class separation line')
#plt.legend(fontsize = 20, loc='upper right')
plt.title('Data points and cluster centroids', fontsize = 22)
plt.xlabel("Cos${\Theta}$", fontsize = 22)
ax = plt.gca()
plt.grid(False)
ax.axes.yaxis.set_visible(False)
plt.xticks(size = 20)
plt.show()


# Homogeneity and completeness
labels_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
labels_pred = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
print('Homogeneity = ', metrics.homogeneity_score(labels_true, labels_pred))
print('Completeness = ', metrics.completeness_score(labels_true, labels_pred))
print('v-score = ', metrics.v_measure_score(labels_true, labels_pred))
print('ARI = ', adjusted_rand_score(labels_true, labels_pred))


# In[18]:


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


validation = np.asarray(validation)


# In[24]:


get_ipython().run_line_magic('matplotlib', '')
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import matplotlib.pyplot as plt
X = np.asarray([total[0][0], total[1][0], total[2][0], total[3][0], total[4][0], total[5][0], total[6][0], total[7][0], total[8][0], total[9][0], total[10][0], total[11][0], total[12][0], total[13][0], total[14][0], total[15][0], total[16][0], total[17][0], total[18][0], total[19][0], total[20][0], total[21][0], total[22][0], total[23][0]])
X = -X
Y = np.asarray([validation[0][0], validation[1][0], validation[2][0], validation[3][0], validation[4][0], validation[5][0]])
Y = -Y
for i in range(0,6):
    X[i] = str(round(X[i], 2))
    print('Stack ', i+1, ': ', X[i])
Y = np.interp(Y, (X.min(), X.max()), (-1, +1))
for i in range(0,6):
    Y[i] = str(round(Y[i], 2))
    print('Stack ', i+1, ': ', Y[i])
Y = np.asarray(Y).reshape(-1, 1)
#print(Y)

ax.tick_params(axis='both', which='major', labelsize=25)
ax.tick_params(axis='both', which='minor', labelsize=25)
# Plotting the cluster centers and the data points on a 2D plane
plt.scatter(Y[0:3, 0], np.zeros(3) + 0, label='Mutant hearts', c='orange', marker='o', s=300)
plt.scatter(Y[3:6, 0], np.zeros(3) + 0, label='Healthy hearts', c='blue', marker='o', s=300)

plt.scatter(kmeans.cluster_centers_[1], np.zeros_like(kmeans.cluster_centers_[1]) + 0, c='green', marker='x', linewidth=5, s=450, label='Healthy centroid')
plt.scatter(kmeans.cluster_centers_[0], np.zeros_like(kmeans.cluster_centers_[0]) + 0, c='black', marker='x', linewidth=5, s=450, label='Mutant centroid')


plt.vlines(0.32, -0.05, 0.05, linestyles='dashed', label = 'Class separation line')
#plt.legend(fontsize = 20, loc='upper right')
plt.title('Data points and cluster centroids', fontsize = 15)
plt.xlabel("Cos${\Theta}$", fontsize = 15)
ax = plt.gca()
plt.grid(False)
ax.axes.yaxis.set_visible(False)
plt.xticks(size = 15)
plt.show()


# In[ ]:




