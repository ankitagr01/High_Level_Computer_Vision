## import packages
import numpy as np
from PIL import Image
from numpy import histogram as hist  # call hist, otherwise np.histogram
from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt

import gauss_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


## function gauss (Question 1.a)

sigma = 4.0
[gx, x] = gauss_module.gauss(sigma)

plt.figure()
plt.plot(x, gx, '.-')
plt.show()


## function gaussianfilter (Question 1.b)
img = rgb2gray(np.array(Image.open('graf.png')))
smooth_img = gauss_module.gaussianfilter(img, sigma)

plt.figure()
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
plt.sca(ax1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.sca(ax2)
plt.imshow(smooth_img, cmap='gray', vmin=0, vmax=255)
plt.show()



## function gaussdx (Question 1.c)

img = np.zeros([27,27])
img[13, 13] = 1.0
plt.figure(), plt.imshow(img, cmap='gray')

sigma = 7.0
[G, x] = gauss_module.gauss(sigma)
[D, x] = gauss_module.gaussdx(sigma)

plt.figure()
plt.plot(x, G, 'b.-')
plt.plot(x, D, 'r-')
plt.legend( ('gauss', 'gaussdx'))
plt.show()

G = G.reshape(1, G.size)
D = D.reshape(1, D.size)

plt.figure()
plt.subplot(2,3,1)
plt.imshow(conv2(conv2(img, G, 'same'), G.T, 'same') , cmap='gray')
plt.subplot(2,3,2)
plt.imshow(conv2(conv2(img, G, 'same'), D.T, 'same') , cmap='gray')
plt.subplot(2,3,3)
plt.imshow(conv2(conv2(img, D.T, 'same'), G, 'same') , cmap='gray')
plt.subplot(2,3,4)
plt.imshow(conv2(conv2(img, D, 'same'), D.T, 'same') , cmap='gray')
plt.subplot(2,3,5)
plt.imshow(conv2(conv2(img, D, 'same'), G.T, 'same') , cmap='gray')
plt.subplot(2,3,6)
plt.imshow(conv2(conv2(img, G.T, 'same'), D, 'same') , cmap='gray')
plt.show()


## function gaussderiv (Question 1.d)

img_c = np.array(Image.open('graf.png')).astype('double')
img = rgb2gray(img_c)
[imgDx, imgDy] = gauss_module.gaussderiv(img, 7.0)

plt.figure()
ax1 = plt.subplot(1,3,1)
ax2 = plt.subplot(1,3,2)
ax3 = plt.subplot(1,3,3)
plt.sca(ax1)
plt.imshow(imgDx, cmap='gray')
plt.sca(ax2)
plt.imshow(imgDy, cmap='gray')
plt.sca(ax3)
imgmag = np.sqrt(imgDx**2 + imgDy**2)
plt.imshow(imgmag, cmap='gray')
plt.show()
