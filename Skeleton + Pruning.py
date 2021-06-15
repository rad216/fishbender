#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tifffile
from skimage import morphology, filters
import cc3d
import skan
import numpy as np
import copy
import matplotlib.pyplot as plt
from PIL import Image
import cv2 

#3D skeletonisation
def skeletonise(input):
    return morphology.skeletonize_3d(input)

#3D pruning
def prune(input,
          sigma: float=0.1,
          consider_end_points: bool=True):

    binary_array = input > 0
    pruned_array = copy.deepcopy(binary_array)

    # Smooth skeleton
    gauss = filters.gaussian(binary_array, sigma=sigma)
    binary_array = morphology.skeletonize_3d(gauss > 0)

    # Find degrees of each pixel
    __, __, degrees = skan.skeleton_to_csgraph(binary_array, unique_junctions=False)

    if consider_end_points:
        degree_one_coords = np.where(degrees == 1)
        print(degree_one_coords)
        most_left = degree_one_coords[2].argmin()
        most_right = degree_one_coords[2].argmax()
        print('Most left', (degree_one_coords[0][most_left], degree_one_coords[1][most_left], degree_one_coords[2][most_left]))
        print('Most right', (degree_one_coords[0][most_right], degree_one_coords[1][most_right], degree_one_coords[2][most_right]))

        degrees[degree_one_coords[0][most_left], degree_one_coords[1][most_left], degree_one_coords[2][most_left]] == 2
        degrees[degree_one_coords[0][most_right], degree_one_coords[1][most_right], degree_one_coords[2][most_right]] == 2
    
    
    # Find individual branch segments which are not branching points
    segments = cc3d.connected_components(np.bitwise_or(degrees == 1, degrees == 2))

    # Iteratively remove segments which end in end point (pixel with degree 1)
    end_points = degrees == 1
    print('Connected components', segments.max() + 1)

    # Find better criterion for removal!!
    for cc in range(segments.max() + 1):
        if np.bitwise_and((segments == cc), end_points).sum() == 1:
            pruned_array = np.bitwise_xor(pruned_array, (segments == cc))

    # Keep longest branch
    cc_pruned_array = cc3d.connected_components(pruned_array)
    counts = np.bincount(cc_pruned_array.flatten())
    # Most common component is background
    counts[counts.argmax()] = 0
    out_array = cc_pruned_array == counts.argmax()

    return out_array

if __name__ == '__main__':

    segmentation_volume = tifffile.imread('D:\\Y4\\Final Year Project\\New datasets\\Healthy To Segment\\Series9++1001\\ManualSegmentationSeries9++1001.tif')
    skeleton = skeletonise(segmentation_volume)
    centreline = prune(skeleton)
    #centreline_int = centreline.astype(int)
    #print(centreline_int)
    #print(centreline[81][196][158].astype(int))
    tifffile.imsave('Series9++NEWprunedSkeleton3D.tiff', centreline.astype(int))


# In[ ]:


# Most left is max row and min col
# Most right is max col and max abs(row)

