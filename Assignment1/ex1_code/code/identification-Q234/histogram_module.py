import numpy as np
from numpy import histogram as hist
import sys

#Add the path where "Gauss_Module.py" file is located to 0th index of Sys.Path
#We need Gauss_Module file to retrieve partial derivatives using Gauss Derivative function
sys.path.insert(0,'../filter-Q1')
import gauss_module

#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    hists = np.zeros(num_bins)
    bins = np.zeros(num_bins)
    bin_len =  255.0 / num_bins

    for i in range(num_bins):
        bins[i] = bin_len * (i + 1)
    bins = np.insert(bins, 0, 0)
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            k = min(int(img_gray[i, j] / bin_len), num_bins)
            hists[k] += 1

    hists, bins = hist(img_gray.reshape(img_gray.size), num_bins, (0,255))
    
    #Normalized
    hists = hists * 1.0 / np.sum(hists)
    print(hists, bins)
    return hists, bins

#  compute joint histogram for each color channel in the image, histogram should be normalized so that sum of all values equals 1
#  assume that values in each channel vary between 0 and 255
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
def rgb_hist(img_color, num_bins):
    assert len(img_color.shape) == 3, 'image dimension mismatch'
    assert img_color.dtype == 'float', 'incorrect image type'

    # define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))
    bin_len = (255.0 / num_bins)

    # execute the loop for each pixel in the image 
    for i in range(img_color.shape[0]):
        for j in range(img_color.shape[1]):
            # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
            kr = min(int(img_color[i, j, 0] / bin_len), num_bins)
            kg = min(int(img_color[i, j, 1] / bin_len), num_bins)
            kb = min(int(img_color[i, j, 2] / bin_len), num_bins)
            hists[kr,kg,kb] += 1
    # normalize the histogram
    hists = hists * 1.0 / np.sum(hists)
    hists = hists.reshape(hists.size)
    print(hists)
    return hists

#  compute joint histogram for r/g values
#  hint: r + g + b = 1 for all the pixels, i.e., r = R / (R + G + B)
#  note that r/g values should be in the range [0, 1];
#  histogram should be normalized so that sum of all values equals 1
#  img_color - input color image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
def rg_hist(img_color, num_bins):
    assert len(img_color.shape) == 3, 'image dimension mismatch'
    assert img_color.dtype == 'float', 'incorrect image type'
    # define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    bin_len = 1.0 / num_bins

    # your code here
    for i in range(img_color.shape[0]):
        for j in range(img_color.shape[1]):
            k = 1.0 * img_color[i, j, :] / sum(img_color[i, j, :])
            kr = min(int(k[0] / bin_len), num_bins-1)
            kg = min(int(k[1] / bin_len), num_bins-1)
            hists[kr, kg] += 1
    hists = hists * 1.0 / np.sum(hists)
    hists = hists.reshape(hists.size)
    return hists

#  compute joint histogram of Gaussian partial derivatives of the image in x and y direction
#  for sigma = 7.0, the range of derivatives is approximately [-30, 30]
#  histogram should be normalized so that sum of all values equals 1
#  img_gray - input grayvalue image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#  note: you can use the function gaussderiv.m from the filter exercise.
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    # compute the first derivatives
    imgderiv = gauss_module.gaussderiv(img_gray, 7.0)

    # quantize derivatives to "num_bins" number of values
    bin_len = 60.0 / num_bins

    # define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    for i in range(imgderiv[0].shape[0]):
        for j in range(imgderiv[0].shape[1]):
            derx = (imgderiv[0][i,j] + 30) // bin_len
            dery = (imgderiv[1][i,j] + 30) // bin_len
            hists[int(derx), int(dery)] += 1
    hists = hists * 1.0 / np.sum(hists)
    hists = hists.reshape(hists.size)
    return hists

def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img1_gray, num_bins_gray, dist_name):
  if dist_name == 'grayvalue':
    return normalized_hist(img1_gray, num_bins_gray)
  elif dist_name == 'rgb':
    return rgb_hist(img1_gray, num_bins_gray)
  elif dist_name == 'rg':
    return rg_hist(img1_gray, num_bins_gray)
  elif dist_name == 'dxdy':
    return dxdy_hist(img1_gray, num_bins_gray)
  else:
    assert 'unknown distance: %s'%dist_name
 