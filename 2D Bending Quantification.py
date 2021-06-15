#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tifffile


# In[3]:


#create a list with the 2D normalised vectors for all stacks in the training dataset
total = []

point_1 = np.array([166,210])
point_2 = np.array([284,297])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([189,203])
point_2 = np.array([287,296])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([178,211])
point_2 = np.array([289,297])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([171,215])
point_2 = np.array([292,298])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([180,163])
point_2 = np.array([297,285])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([177,184])
point_2 = np.array([309,285])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([189,164])
point_2 = np.array([294,305])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([170,194])
point_2 = np.array([300,270])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([181,192])
point_2 = np.array([298,299])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([177,187])
point_2 = np.array([282,295])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([172,181])
point_2 = np.array([275,285])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([175,182])
point_2 = np.array([285,279])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([161,204])
point_2 = np.array([281,289])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([169,197])
point_2 = np.array([286,299])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([164,243])
point_2 = np.array([301,291])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([164,257])
point_2 = np.array([290,293])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([155,248])
point_2 = np.array([290,296])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([167,254])
point_2 = np.array([282,298])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([170,230])
point_2 = np.array([295,280])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([168,242])
point_2 = np.array([295,282])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([196,211])
point_2 = np.array([326,264])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([166,244])
point_2 = np.array([288,288])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([165,251])
point_2 = np.array([301,291])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

point_1 = np.array([184,217])
point_2 = np.array([289,312])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
total.append(vec_norm)

total = np.asarray(total)


# In[4]:


#print the normalised 2D vectors
for i in range(0,24):
    total[i][0] = str(round(total[i][0], 2))
    total[i][1] = str(round(total[i][1], 2))
    print('Stack ', i+1, ': ', total[i][0],',', total[i][1])


# In[5]:


get_ipython().run_line_magic('matplotlib', '')
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import matplotlib.pyplot as plt
import math

for_validation = total
total =  np.interp(total, (total.min(), total.max()), (-1, +1))
kmeans = KMeans(n_clusters=2, random_state=0).fit(total)
for i in range(0,24):
   total[i][0] = str(round(total[i][0], 2))
   total[i][1] = str(round(total[i][1], 2))
   print('Stack ', i+1, ': ', total[i][0],',', total[i][1])
# Get the cluster centroids
print('centroids: ', kmeans.cluster_centers_)
# Get the cluster labels
print(kmeans.labels_)

mutants1 = total[0:7]
mutants2 = total[8:14]
mutants = np.vstack((mutants1,mutants2))
healthy = total[14:23]



fig = plt.figure()
plt.tick_params(axis='both', which='major', labelsize=35)
plt.tick_params(axis='both', which='minor', labelsize=35)

plt.scatter(healthy[:,0], healthy[:,1], marker='o', c='blue', s=300, label='Healthy hearts')
plt.scatter(mutants[:,0], mutants[:,1], marker='o', c='orange', s=300, label='Mutant hearts')
plt.scatter(total[23,0], total[23,1], marker='o', c='orange', edgecolors='red', linewidth=3, s=300, label='Misclassified healthy hearts')
plt.scatter(total[7,0], total[7,1], marker='o', c='blue', edgecolors='red', linewidth=3, s=300, label='Misclassified mutant hearts')

plt.scatter(kmeans.cluster_centers_[0,0],  kmeans.cluster_centers_[0,1], c='black', marker='x', linewidth=5, s=450, label='Mutant centroid')
plt.scatter(kmeans.cluster_centers_[1,0],  kmeans.cluster_centers_[1,1],c='green', marker='x', linewidth=5, s=450, label='Healthy centroid')
#plt3d.title('Data points and cluster centroids', fontsize = 22)
plt.grid(False)
point1 = [-0.92366409, -0.12636082]
point2 = [-0.36928094, 0.78859223]
x_values = [point1[0], point2[0]]
y_values = [point1[1], point2[1]]
plt.plot(x_values, y_values, 'k--')
plt.legend(fontsize = 20, loc='upper right', bbox_to_anchor=(2.5, 0.5))


plt.show()
distance = math.sqrt((kmeans.cluster_centers_[0][0]-kmeans.cluster_centers_[1][0])**2 + (kmeans.cluster_centers_[0][1]-kmeans.cluster_centers_[1][1])**2)
print('distance: ', distance)

labels_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
labels_pred = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
print('Homogeneity = ', metrics.homogeneity_score(labels_true, labels_pred))
print('Completeness = ', metrics.completeness_score(labels_true, labels_pred))
print('v-score = ', metrics.v_measure_score(labels_true, labels_pred))
print('ARI = ', adjusted_rand_score(labels_true, labels_pred))


# In[6]:


validation = []

point_1 = np.array([204,198])
point_2 = np.array([273,325])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
validation.append(vec_norm)

point_1 = np.array([129,172])
point_2 = np.array([245,282])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
validation.append(vec_norm)

point_1 = np.array([141,195])
point_2 = np.array([249,331])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
validation.append(vec_norm)

point_1 = np.array([150,226])
point_2 = np.array([293,298])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
validation.append(vec_norm)

point_1 = np.array([144,182])
point_2 = np.array([297,245])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
validation.append(vec_norm)

point_1 = np.array([161,215])
point_2 = np.array([281,266])
vec = point_1 - point_2
norm = np.linalg.norm(vec)
vec_norm = vec/norm
validation.append(vec_norm)
print(validation)

validation = np.asarray(validation)
validation =  np.interp(validation, (for_validation.min(), for_validation.max()), (-1, +1))
print('interpolated: ', validation)
fig = plt.figure()

plt.tick_params(axis='both', which='major', labelsize=25)
plt.tick_params(axis='both', which='minor', labelsize=25)
plt.scatter(validation[0:3,0], validation[0:3,1], marker='o', c='orange', s=300, label='Mutant hearts')
plt.scatter(validation[3:6,0], validation[3:6,1], marker='o', c='blue', s=300, label='Healthy hearts')

plt.scatter(kmeans.cluster_centers_[0,0],  kmeans.cluster_centers_[0,1], c='black', marker='x', linewidth=5, s=450, label='Mutant centroid')
plt.scatter(kmeans.cluster_centers_[1,0],  kmeans.cluster_centers_[1,1],c='green', marker='x', linewidth=5, s=450, label='Healthy centroid')
plt.legend(fontsize = 20, loc='upper right')
x_values = [point1[0], point2[0]]
y_values = [point1[1], point2[1]]
plt.plot(x_values, y_values, 'k--')
plt.grid(False)
plt.show()


# In[ ]:




