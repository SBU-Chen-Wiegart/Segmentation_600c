#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:59:14 2021

@author: karenchen-wiegart
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import io
from skimage import morphology
import time
from skimage import feature
from scipy.signal import find_peaks

dir = '/home/karenchen-wiegart/ChenWiegartgroup/Xiaoyin/53609_53624/'
fn = '53610_alignby53609.tif'

img0 = io.imread(dir+fn)

slice_start = 32

# In[0]
'''Function'''
def num_largest_area(labels_, num_labels_):
    l=list()
    for i in range(1, num_labels_):
        index = np.argwhere(labels_==i)
        l.append(len(index))
        # print(l)
    return l.index(max(l))+1
        
# In[1]
'''Threshold Segmentation'''
num_slice = img0.shape[0]
img0_seg = img0.copy()
for i in range(slice_start, num_slice):

    img = img0[i]
    a = img.flatten()
    hist, bins = np.histogram(a, bins=10, range=[0.0001, 5])
    
    peaks, _ = find_peaks(hist)
    if len(peaks)==2:
        hist_btw2peaks = hist[peaks[0]:peaks[1]]
    else:
        print('length of peaks is larger than 3.')
    hist_min = np.amin(hist_btw2peaks)
    index = np.where(hist==hist_min)
    threshold = bins[index]
    # plt.figure(figsize=(20,15))
    # plt.plot(bins[:-1], hist)
    # plt.scatter(threshold, hist_min, marker='x', c='r', s=200)
    
    seg = np.where(img>threshold, 1, 0).astype(np.float32)
    s = ndi.generate_binary_structure(2, 2)
    labels, num_labels = ndi.label(seg, structure=s)
    temp = num_largest_area(labels, num_labels)
    seg_mainbody = np.where(labels==temp, 1, 0).astype(np.float32)
    img0_seg[i] = seg_mainbody
    
io.imsave('seg_53610.tiff', img0_seg)

# c = num_largest_area(labels, num_labels)

# plt.figure(figsize=(20,15))
# plt.imshow(labels)

# plt.figure(figsize=(20,15))
# plt.imshow(img)

# In[2]
'''Calculate interface area'''
img0_seg = io.imread('seg_53610.tiff')
num_slices = img0_seg.shape[0]
area = list()
for i in range(slice_start, num_slices):
    seg_mainbody = img0_seg[i]
    distance = ndi.distance_transform_edt(seg_mainbody)
    index = np.argwhere(distance==1)
    area_t = len(index)
    area.append(area_t)

total_area = sum(area)
print(total_area)
# In[3]
'''Supplementary information'''
seg_mainbody = img0_seg[150]
plt.figure(figsize=(20,15))
plt.imshow(seg_mainbody)

distance = ndi.distance_transform_edt(seg_mainbody)
index = np.argwhere(distance==1)
area_interface = len(index)




edges = feature.canny(seg_mainbody, sigma=1).astype(np.float32)
plt.figure(figsize=(20,15))
plt.imshow(edges)
plt.colorbar()

plt.figure(figsize=(20,15))
plt.imshow(distance)

plt.figure(figsize=(20,15))
plt.imshow(img0_seg[50])

plt.figure(figsize=(20,15))
plt.imshow(img0_seg[150])

s = ndi.generate_binary_structure(2, 3)
edges_filled = ndi.binary_fill_holes(edges)

seg_mainbody_filled = ndi.binary_fill_holes(seg_mainbody)
plt.figure(figsize=(20,15))
plt.imshow(seg_32_300[266])

seg_32_300 = img0_seg[32:299]
io.imsave('seg_32_300.tiff', seg_32_300)
