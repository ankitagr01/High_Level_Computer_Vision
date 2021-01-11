import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

#
# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image
#

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

  hist_isgray = histogram_module.is_grayvalue_hist(hist_type)

  model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
  query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)

  D = np.zeros((len(model_images), len(query_images)))
  # best_match = [99999] * len(query_images)
  # your code here
  # compute distance matrix
  for i in range(len(model_images)):
      hist1 = model_hists[i]
      for j in range(len(query_images)):
          hist2 = query_hists[j]
          D[i, j] = dist_module.get_dist_by_name(hist1, hist2, dist_type)
          # if D[i,j] < best_match[j]:
          #     best_match[j] = D[i,j]

  # find the best match for each query image
  best_match = D.argmin(axis=0)
  # print (best_match)

  return best_match, D

def compute_histograms(image_list, hist_type, hist_isgray, num_bins):

  image_hist = []

  # compute hisgoram for each image and add it at the bottom of image_hist
  # your code here
  for i in range(len(image_list)):
      images = np.array(Image.open(image_list[i]))
      images = images.astype('double')

      if hist_isgray:
          image_gray = rgb2gray(images)
          hist = histogram_module.get_hist_by_name(image_gray, num_bins, hist_type)
      else:
          hist = histogram_module.get_hist_by_name(images, num_bins, hist_type)

      # print(len(hist))
      if len(hist) == 2 and len(hist[0]) > 1:
          hist = hist[0]

      image_hist.append(hist)

  return image_hist

#
# for each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# note: use the previously implemented function 'find_best_match'
# note: use subplot command to show all the images in the same Python figure, one row per query image
#

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):

  plt.figure()

  num_nearest = 5  # show the top-5 neighbors

  # your code here
  #using find_bset_match to get D
  [best_match, D] = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)

  index = D.argsort(axis=0)
  qlen = len(query_images)

  for i in range(qlen):
      plt.subplot(qlen, num_nearest + 1, i * (num_nearest + 1) + 1)
      plt.imshow(np.array(Image.open(query_images[i])))

      for j in range(num_nearest):
          plt.subplot(qlen, num_nearest + 1, i * (num_nearest + 1) + j + 2)
          plt.imshow(np.array(Image.open(model_images[index[j, i]])))

  plt.show()



