# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 09:51:13 2021

@author: Chen-Wiegart
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed, expand_labels
from skimage import segmentation, filters
from skimage.feature import peak_local_max
from skimage import io
from skimage import morphology
import time
from skimage import feature
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from joblib import Parallel, delayed

list_data = [x for x in range(53578, 53626)]
del list_data[1:3]

list_data = [53593]

dir_mask = r'D:\Xiaoyin\53578-53624_crop_seg_mask'
dir_img = r'D:\Xiaoyin\53578-53624crop'

dir_out = r'D:\Xiaoyin\53578-53624_crop_seg_mask_histogram'

dir_raw_cropped = r'D:\Xiaoyin\53578-53624_raw_cropped_by_mask'

dir_out_seg = r'D:\Xiaoyin\53578-53624_cop_seg_mask_seg'

# list_data=[53592]

# In[1]
def first_derivative(l):
    der1 = list()
    for i in range(len(l)-1):
        der1.append(l[i+1]-l[i])
    return der1

def func(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

def label_fill_holes(label_buffer, i):

    buffer_i = label_buffer==i
    s2 = ndi.generate_binary_structure(3, 1)    
    dilate = ndi.binary_dilation(buffer_i, structure=s2)
    surr = img_expand[dilate & ~buffer_i]
    num_surr_particle = sum(surr==1)
    num_surr_bkg = sum(surr==2)
    if num_surr_particle >= num_surr_bkg:
        img_expand[buffer_i] = 2
    else:
        img_expand[buffer_i] = 1
    if i % 100==0:
        print(i)

# In[2]

for i in list_data:
    fn_mask = r'\{}_crop_seg_mask.tiff'.format(i)
    fn_img = r'\{}.tiff'.format(i)
    mask = io.imread(dir_mask+fn_mask)
    # plt.figure()
    # plt.imshow(mask)
    struct1 = ndi.generate_binary_structure(2,1)
    mask = ndi.binary_dilation(mask, structure=struct1, iterations=4)
    # plt.figure()
    # plt.imshow(mask)
    img = io.imread(dir_img+fn_img)
    mask_3d = np.zeros((300, 685, 685))
    mask_3d[:] = mask
    index = np.argwhere(mask_3d==0)
    img[tuple(index.T)] = 0
    img_crop = img[60:260]
    # io.imsave(dir_raw_cropped+'\{}.tiff'.format(i), img_crop)
    # print('{} is finished'.format(i))
    flat = img_crop.flatten()
    hist, bins = np.histogram(flat, bins = 256, range=[0.0001, 10])
    plt.figure()
    plt.plot(bins[:-1], hist)
    peak, pi = find_peaks(hist,height=600000)
    '''hist_gaussian is the histogram whose left peak part is a flip of the right peak part.
        This can be assumed as the histogarm of particles only.'''
    hist_gaussian = hist.copy()    # hist_gaussian = hist of particle only
    temp = hist[peak[0]:2*peak[0]]
    hist_gaussian[:peak[0]] = temp[::-1]
    plt.plot(bins[:-1], hist_gaussian)
    
    diff = hist-hist_gaussian    # diff = hist of bkg only

    plt.plot(bins[:-1], diff)
    peak_diff, pi = find_peaks(diff,height=150000)
    diff_gaussian = diff[:2*peak_diff[0]]    # crop part of hist of bkg to fit
    diff_gaussian_bins = bins[:2*peak_diff[0]]    # crop part of hist of bkg to fit
    popt, pcov = curve_fit(func, diff_gaussian_bins, diff_gaussian)
    # plt.figure()
    plt.plot(diff_gaussian_bins, func(diff_gaussian_bins, *popt), 'g--',
          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.xlabel('Pixel value')
    plt.ylabel('Counts')
    plt.legend(['Original histogram', 'histogram of sample', 'histogram of bkg', 'fitted histogram of bkg'])
    # plt.plot(bins[:-1], diff)
    # plt.plot(diff_gaussian_bins, diff_gaussian, 'r--')
    '''The intersection of two histograms, the one of particle and the one of bkg, will be
        defined as the threshold.'''
    # Intersection
    # diff_2histogram = func(diff_gaussian_bins, *popt)-hist_gaussian[:2*peak_diff[0]]
    # index = np.argwhere(diff_2histogram>0)
    # threshold = (bins[len(index)]+bins[len(index)-1])/2
    # print(threshold)

    peak_particle_bins = peak[0]    # peak_particle is the position of bins, peak_particle=45
    peak_bkg_bins = max(np.argwhere(bins<popt[1]))[0]
    
    area_particle = sum(hist[peak_particle_bins:])
    area_bkg = sum(hist[:peak_bkg_bins])
    ratio = area_particle/area_bkg
    sum_btwpeaks = sum(hist[peak_bkg_bins:peak_particle_bins])
    
    area = 0
    index = peak_bkg_bins
    while area<1/(ratio+1)*sum_btwpeaks:
        area+=hist[index]
        index+=1
    
    '''Expand labels'''
    # 1=particel, 2=bkg, 0=buffer area
    img_label = img_crop.copy()
    # Seeding area for particle
    threshold_particle = bins[index-1]
    index_particle = np.argwhere(img_crop>threshold_particle)
    img_label[tuple(index_particle.T)] = 1
    # Seeding area for bkg
    threshold_bkg = bins[peak_bkg_bins]
    index_bkg = np.argwhere((img_crop>0) & (img_crop<threshold_bkg))
    img_label[tuple(index_bkg.T)] = 2
    # Seeding area for buffer area
    index_buffer = np.argwhere((img_crop>=threshold_bkg) & (img_crop<=threshold_particle))
    img_label[tuple(index_buffer.T)] = 0
    
    # edges = feature.canny(img_crop[150], sigma=5)
    global img_expand
    img_expand = expand_labels.expand_labels(img_label, 1.5)
    plt.figure()
    plt.imshow(img_expand[100])
    '''Fill holes with majority surrounding values'''
    img_expand[mask_3d[60:260]==0] = -1
    # plt.figure()
    # plt.imshow(img_expand[100])
    img_buffer = img_expand==0
    s2 = ndi.generate_binary_structure(3, 1)    
    label_buffer, num_buffer = ndi.label(img_buffer, structure=s2)

    # img_seg = np.where(img_crop>threshold_particle, 100, 50)
    # io.imsave(dir_out_seg+'\{}_test.tiff'.format(i), np.float32(img_seg))    
    #

# plt.figure()
# plt.imshow(mask_3d[100])

# plt.figure()
# plt.imshow(img[100]>1.62)
# plt.figure()
# plt.plot(diff_2histogram)


# plt.figure()
# plt.imshow(img_crop[100])

# plt.figure()
# plt.imshow(img_label[100])

plt.figure()
plt.imshow(img_buffer[100])


# plt.figure()
# plt.imshow(img_buffer[100])
# edges = edges*10
# img_buffer = np.zeros(img_crop.shape)
# img_buffer[tuple(index_buffer.T)] = 5
# plt.figure()
# # plt.imshow(edges)
# plt.imshow(img_buffer[150]-edges)

# plt.figure()
# plt.plot(bins[:-2], aa)

# a1 = sum(hist[peak_bkg_bins:60])
# a2 = sum(hist[60:peak_particle_bins])

# d=r'D:\Xiaoyin\53578-53624_crop_seg'
# f=r'\53592_crop_seg.tiff'

# img0=io.imread(d+f)
# img_s=np.sum(img0, axis=0)
# plt.figure()
# plt.imshow(img_s)
# plt.colorbar()
 