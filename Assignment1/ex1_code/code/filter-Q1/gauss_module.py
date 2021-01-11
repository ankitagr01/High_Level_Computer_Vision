
# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
from scipy import signal
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def gauss(sigma):
    # ...
    Gx = np.array([])
    x  = np.array([])
    for val in range(math.ceil(-3*sigma),math.floor(3*sigma+1)):
        Gx=np.append(Gx,[(1/sigma/math.sqrt(2*math.pi))*math.exp(-val**2/(2*sigma**2))])
        x= np.append(x,[val])
    
    return Gx, x

def gaussderiv(img, sigma):
    # ...
    D = gaussdx(sigma)[0]
    G = gauss(sigma)[0]
    imgDx = np.zeros(img.shape)
    imgDy = np.zeros(img.shape)
    for i in range(np.size(img,0)):
        imgDx[i,:] = np.convolve(img[i,:], D, 'same')
        imgDy[i,:] = np.convolve(img[i,:], G, 'same')
    for i in range(np.size(img,1)):
        imgDy[:,i] = np.convolve(imgDy[:,i], D, 'same')
        imgDx[:,i] = np.convolve(imgDx[:,i], G, 'same')

    return imgDx, imgDy

def gaussdx(sigma):
    #
    D = np.array([])
    x = np.array([])
    for val in range(math.ceil(-3*sigma),math.floor(3*sigma+1)):
        D=np.append(D,[(-val/(sigma**3)/math.sqrt(2*math.pi))*math.exp(-val**2/(2*sigma**2))])
        x= np.append(x,[val])
    return D, x


def gaussianfilter(img, sigma):
    # ...
    kernal_1d= gauss(sigma)[0]
    outimage=np.zeros(img.shape)
    
    for i in range(np.size(img,1)):
        outimage[:,i] = np.convolve(img[:,i], kernal_1d, 'same')
    for i in range(np.size(img,0)):
        outimage[i,:] = np.convolve(outimage[i,:], kernal_1d, 'same')
    
    return outimage



