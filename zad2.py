# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 22:12:05 2021

@author: mm180261d
"""

from skimage import color
from skimage import *
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from pylab import *
from scipy import ndimage


miner = imread('miner.jpg')
plt.figure()
io.imshow(miner)
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
print(laplacian)
minerYUV = color.rgb2yuv(miner)
compY = minerYUV[:,:,0]
plt.figure()
io.imshow(compY, cmap='gray')

filtratedCompY = ndimage.correlate(compY, laplacian)
filtratedCompY[filtratedCompY<0] = 0
filtratedCompY[filtratedCompY>1] = 1
minerYUV[:, :, 0] = compY - filtratedCompY
minerYUV[minerYUV[:,:,0]>1,0] = 1
minerYUV[minerYUV[:,:,0]<0,0] = 0
plt.figure()
io.imshow(minerYUV[:, :, 0], cmap='gray')

minerOut = color.yuv2rgb(minerYUV)
plt.figure()
io.imshow(minerOut)

minerOut[minerOut>1] = 1
minerOut = img_as_ubyte(minerOut)

imsave('miner_sharp.jpg', minerOut)