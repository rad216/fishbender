#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tifffile
im = tifffile.imread('D:\\Y4\\Final Year Project\\Napari\\Centreline\\Data for Fiji\\Warped Points New Orientation\\Tiff Files\\Series4WrapedCentreline.tif').astype('bool')


# In[ ]:


x_coord = 292
y_coord = 298


# In[ ]:


def find_z_coordinate(arr, x_coord, y_coord, val):
    idx = np.where(arr == val)
    return int(idx[0][np.intersect1d(np.where(idx[1] == y_coord), np.where(idx[2] == x_coord))])


# In[ ]:


find_z_coordinate(im, x_coord, y_coord, True)

