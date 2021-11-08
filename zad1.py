# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from pylab import *

import matplotlib.pyplot as plt
import skimage
from skimage import color
from skimage import exposure
from skimage import filters
from skimage import io

from scipy import ndimage

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import numpy as np

def dressQueenHSVGauss(rgbQueen: np.array, rgbImg: np.array) -> np.array:
    imgQueenHSV = color.rgb2hsv(rgbQueen)
    # imgHSV = color.rgb2hsv(rgbImg)
    imgYUV = color.rgb2yuv(rgbImg)
    imgQueenYUV = color.rgb2yuv(rgbQueen)
    
    fig, ax = plt.subplots(2, 2, figsize=(12,9), dpi=80);
    tight_layout();
    ax[0,0].imshow(rgbQueen); ax[0,0].set_title('RGB', fontsize=14);
    ax[0,1].imshow(imgQueenHSV[:,:,2], vmin=0, vmax=1, cmap='gray'); ax[0,1].set_title('V', fontsize=14);
    ax[1,0].imshow(imgQueenHSV[:,:,1], vmin=0, vmax=1, cmap='jet'); ax[1,0].set_title('S', fontsize=14);
    ax[1,1].imshow(imgQueenHSV[:,:,0], vmin=0, vmax=1, cmap='jet'); ax[1,1].set_title('H', fontsize=14);
    
    mask = (imgQueenHSV[:,:,0]>0.4) & (imgQueenHSV[:,:,0]<0.6) & (imgQueenHSV[:,:,1]>0.5)
    mask1 = mask
    
    mask = mask*1.
    mask_gauss = filters.gaussian(mask, sigma=1, truncate=5)
    
    newU = np.ones(rgbQueen[:,:,0].shape)*imgYUV[0,0,1]
    newU[416:832,200:616] = imgYUV[:,:,1]
    
    newV = np.ones(rgbQueen[:,:,0].shape)*imgYUV[0,0,2]
    newV[416:832,200:616] = imgYUV[:,:,2]
    
    mask1 = mask_gauss>0.1
    
    outputU = mask1*newU
    outputV = mask1*newV
    
    imgQueenYUV[mask1,1] = outputU[mask1]
    imgQueenYUV[mask1,2] = outputV[mask1]
    
    imgQueenOutput = color.yuv2rgb(imgQueenYUV)
    
    plt.figure()
    io.imshow(imgQueenOutput)
    return imgQueenOutput


if __name__=="__main__":
    imgQueen = imread('queen_dress.jpg')
    imgRacism = imread('noToRacism.jpg')
    
    io.imshow(imgQueen)
    
    # dressQueenLCH(imgQueen, imgRacism)
    
    dressQueenHSVGauss(imgQueen, imgRacism)
    
    # dressQueenHSVBox(imgQueen, imgRacism)