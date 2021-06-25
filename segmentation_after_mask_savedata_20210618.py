# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 09:51:13 2021

@author: Chen-Wiegart
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
from scipy.optimize import curve_fit

list_data = [x for x in range(53578, 53626)]
del list_data[1:3]

dir_mask = r'D:\Xiaoyin\53578-53624_crop_seg_mask'
dir_img = r'D:\Xiaoyin\53578-53624crop'

dir_out = r'D:\Xiaoyin\53578-53624_crop_seg_mask_histogram'

dir_raw_cropped = r'D:\Xiaoyin\53578-53624_raw_cropped_by_mask'

# list_data=[53592]

# In[1]
def first_derivative(l):
    der1 = list()
    for i in range(len(l)-1):
        der1.append(l[i+1]-l[i])
    return der1

def func(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))
    

# In[2]

for i in list_data:
    fn_mask = r'\{}_crop_seg_mask.tiff'.format(i)
    fn_img = r'\{}.tiff'.format(i)
    mask = io.imread(dir_mask+fn_mask)
    img = io.imread(dir_img+fn_img)
    mask_3d = np.zeros((300, 685, 685))
    mask_3d[:] = mask
    index = np.argwhere(mask_3d==0)
    img[tuple(index.T)] = 0
    img_crop = img[60:260]
    io.imsave(dir_raw_cropped+'\{}.tiff'.format(i), img_crop)
    print('{} is finished'.format(i))
    # flat = img_crop.flatten()
    # hist, bins = np.histogram(flat, bins = 256, range=[0.0001, 10])
    # plt.figure()
    # plt.plot(bins[:-1], hist)
    # plt.savefig(dir_out+'\{}.png'.format(i))
    # peak, pi = find_peaks(hist,height=600000)
    # peak = peak[0]
    # hist_gaussian = hist.copy()
    # temp = hist[peak:2*peak]
    # hist_gaussian[:peak] = temp[::-1]
    # plt.plot(bins[:-1], hist_gaussian)
    # diff = hist-hist_gaussian
    # # plt.figure()
    # plt.plot(bins[:-1], diff)
    # peak_diff, pi = find_peaks(diff,height=150000)
    # diff_gaussian = diff[:2*peak_diff[0]]
    # diff_gaussian_bins = bins[:2*peak_diff[0]]
    # popt, pcov = curve_fit(func, diff_gaussian_bins, diff_gaussian)
    # # plt.figure()
    # plt.plot(diff_gaussian_bins, func(diff_gaussian_bins, *popt), 'g--',
    #       label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # # plt.plot(bins[:-1], diff)
    # # plt.plot(diff_gaussian_bins, diff_gaussian, 'r--')
    # threshold = 2*popt[1]*0.8
    # img_seg = np.where(img>threshold, 1, 0)
    # io.imsave()    

# plt.figure()
# plt.imshow(mask_3d[100])

# plt.figure()
# plt.imshow(img[100]>1.62)


# plt.figure()
# plt.imshow(img[100])

# plt.figure()
# plt.plot(bins[:-2], aa)