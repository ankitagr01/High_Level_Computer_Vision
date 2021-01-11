## import packages
import numpy as np
from PIL import Image
from numpy import histogram as hist  # call hist, otherwise np.histogram
import matplotlib.pyplot as plt

import histogram_module
import dist_module
import match_module
import rpc_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


## gray-value histograms (Question 2.a)

img_color = np.array(Image.open('./model/obj100__0.png'))
img_gray = rgb2gray(img_color.astype('double'))

plt.figure()
plt.subplot(1,3,1)
plt.imshow(img_color)

plt.subplot(1,3,2)
num_bins_gray = 40
hist_gray1, bin_gray1 = hist(img_gray.reshape(img_gray.size), num_bins_gray,(0,255))
plt.bar((bin_gray1[0:-1] + bin_gray1[1:])/2, hist_gray1)

plt.subplot(1,3,3)
hist_gray2, bin_gray2 = histogram_module.normalized_hist(img_gray, num_bins_gray)
plt.bar((bin_gray2[0:-1] + bin_gray2[1:])/2, hist_gray2)
plt.show()

## more histograms (Question 2.b)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_color)

num_bins_color = 5
plt.subplot(1,2,2)
hist_rgb1 = histogram_module.rgb_hist(img_color.astype('double'), num_bins_color)
plt.bar(np.array(range(1,hist_rgb1.size+1)),hist_rgb1)
plt.show()

# compose rg_hist.m and dxdy_hist.m as well

# add your code here



## distance functions (Question 2.c)

image_files1 = ['./model/obj1__0.png']
image_files2 = ['./model/obj91__0.png', './model/obj94__0.png']

plt.figure()
plt.subplot(1,3,1); plt.imshow(np.array(Image.open(image_files1[0])), vmin=0, vmax=255); plt.title(image_files1[0])
plt.subplot(1,3,2); plt.imshow(np.array(Image.open(image_files2[0])), vmin=0, vmax=255); plt.title(image_files2[0])
plt.subplot(1,3,3); plt.imshow(np.array(Image.open(image_files2[1])), vmin=0, vmax=255); plt.title(image_files2[1])
plt.show()


print('distance functions:')
distance_types = ['l2', 'intersect', 'chi2']
print(distance_types)
print('\n')

print('histogram types:')
hist_types = ['grayvalue', 'rgb', 'rg', 'dxdy']
print(hist_types)
print('\n')

num_bins_color = 30;
num_bins_gray = 90;

for imgidx1 in range(len(image_files1)):
  img1_color = np.array(Image.open(image_files1[imgidx1]))
  img1_gray = rgb2gray(img1_color.astype('double'))
  img1_color = img1_color.astype('double')

  for imgidx2 in range(len(image_files2)):
    img2_color = np.array(Image.open(image_files2[imgidx2]))
    img2_gray = rgb2gray(img2_color.astype('double'))
    img2_color = img2_color.astype('double')

    D = np.zeros( (len(distance_types), len(hist_types)) )

    for didx in range(len(distance_types)):

      for hidx in range(len(hist_types)):



        if histogram_module.is_grayvalue_hist(hist_types[hidx]):
          hist1 = histogram_module.get_hist_by_name(img1_gray, num_bins_gray, hist_types[hidx])
          hist2 = histogram_module.get_hist_by_name(img2_gray, num_bins_gray, hist_types[hidx])

          if len(hist1) == 2 and len(hist1[0]) > 1:
            hist1 = hist1[0]
          if len(hist2) == 2 and len(hist2[0]) > 1:
            hist2 = hist2[0]

          #there was error here as we declared in 92 first dist_type than hist_type
          D[didx, hidx] = dist_module.get_dist_by_name(hist1, hist2, distance_types[didx])
        else:
          hist1 = histogram_module.get_hist_by_name(img1_color, num_bins_color, hist_types[hidx])
          hist2 = histogram_module.get_hist_by_name(img2_color, num_bins_color, hist_types[hidx])

          if len(hist1) == 2 and len(hist1[0]) > 1:
            hist1 = hist1[0]
          if len(hist2) == 2 and len(hist2[0]) > 1:
            hist2 = hist2[0]

          D[hidx, didx] = dist_module.get_dist_by_name(hist1, hist2, distance_types[didx])

    print('compare image "%s" to "%s":'% (image_files1[imgidx1], image_files2[imgidx2]))
    print(D)
    print('\n')



## Find best match (Question 3.a)

with open('model.txt') as fp:
    model_images = fp.readlines()
model_images = [x.strip() for x in model_images] 

with open('query.txt') as fp:
    query_images = fp.readlines()
query_images = [x.strip() for x in query_images] 

eval_dist_type = 'intersect';
eval_hist_type = 'rg';
eval_num_bins = 30;


[best_match, D] = match_module.find_best_match(model_images, query_images, eval_dist_type, eval_hist_type, eval_num_bins)


## visualize nearest neighbors (Question 3.b)
query_images_vis = [query_images[i] for i in np.array([0,4,9])]
match_module.show_neighbors(model_images, query_images_vis, eval_dist_type, eval_hist_type, eval_num_bins)


## compute recognition percentage (Question 3.c)
# import ipdb; ipdb.set_trace()
num_correct = sum( best_match == range(len(query_images)) )
print('number of correct matches: %d (%f)\n'% (num_correct, 1.0 * num_correct / len(query_images)))


## plot recall_precision curves (Question 4)

with open('model.txt') as fp:
    model_images = fp.readlines()
model_images = [x.strip() for x in model_images] 

with open('query.txt') as fp:
    query_images = fp.readlines()
query_images = [x.strip() for x in query_images] 

eval_num_bins = 10;


plt.figure()
rpc_module.compare_dist_rpc(model_images, query_images, ['chi2', 'intersect', 'l2'], 'rg', eval_num_bins, ['r', 'g', 'b'])
plt.title('RG histograms')
plt.show()


plt.figure()
rpc_module.compare_dist_rpc(model_images, query_images, ['chi2', 'intersect', 'l2'], 'rgb', int(eval_num_bins / 2), ['r', 'g', 'b'])
#added int as eval_num_bins was in float and we need integer.
plt.title('RGB histograms')
plt.show()

plt.figure()
rpc_module.compare_dist_rpc(model_images, query_images, ['chi2', 'intersect', 'l2'], 'dxdy', eval_num_bins, ['r', 'g', 'b'])
plt.title('dx/dy histograms')
plt.show()

