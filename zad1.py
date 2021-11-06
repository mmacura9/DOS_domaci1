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

def dressQueenLCH(rgbQueen: np.array, rgbImg: np.array) -> np.array:
    imgQueenLab = color.rgb2lab(rgbQueen)
    imgQueenLCh = color.lab2lch(imgQueenLab)
    
    Labimg = color.rgb2lab(rgbImg)
    LChimg = color.lab2lch(Labimg)
    
    fig, ax = plt.subplots(2, 2, figsize=(12,9), dpi=80);
    tight_layout();
    ax[0,0].imshow(rgbQueen); ax[0,0].set_title('RGB', fontsize=14);
    ax[0,1].imshow(imgQueenLCh[:,:,0], vmin=np.min(imgQueenLCh[:,:,0]), vmax=np.max(imgQueenLCh[:,:,0]), cmap='gray'); ax[0,1].set_title('L', fontsize=14);
    ax[1,0].imshow(imgQueenLCh[:,:,1], vmin=np.min(imgQueenLCh[:,:,1]), vmax=np.max(imgQueenLCh[:,:,1]), cmap='jet'); ax[1,0].set_title('C', fontsize=14);
    ax[1,1].imshow(imgQueenLCh[:,:,2], vmin=0, vmax=2*pi, cmap='jet'); ax[1,1].set_title('h', fontsize=14);
    
    downTreshold = 3*pi/4
    upTreshold = 5*pi/4
    mask = (imgQueenLCh[:,:,2]>downTreshold) & (imgQueenLCh[:,:,2]<upTreshold)
    mask1 = mask
    
    mask = mask*1.
    mask_gauss = filters.gaussian(mask, sigma=10, truncate=5)
    
    newH = np.ones(rgbQueen[:,:,0].shape)*LChimg[0,0,2]
    newH[416:832,200:616] = LChimg[:,:,2]
    
    newC = np.ones(rgbQueen[:,:,0].shape)*LChimg[0,0,1]
    newC[416:832,200:616] = LChimg[:,:,1]
    
    mask1 = mask_gauss>0.5
    
    imgQueenLCh[:,:,2] = mask1*newH + ~mask1*imgQueenLCh[:,:,2]
    imgQueenLCh[:,:,1] = mask1*newC + ~mask1*imgQueenLCh[:,:,1]
    
    imgQueenLab = color.lch2lab(imgQueenLCh)
    imgQueenOutput = color.lab2rgb(imgQueenLab)
    fig = plt.figure()
    io.imshow(imgQueenOutput)
    
    return imgQueenOutput

def dressQueenHSVGauss(rgbQueen: np.array, rgbImg: np.array) -> np.array:
    imgQueenHSV = color.rgb2hsv(rgbQueen)
    imgHSV = color.rgb2hsv(rgbImg)
    
    fig, ax = plt.subplots(2, 2, figsize=(12,9), dpi=80);
    tight_layout();
    ax[0,0].imshow(rgbQueen); ax[0,0].set_title('RGB', fontsize=14);
    ax[0,1].imshow(imgQueenHSV[:,:,2], vmin=0, vmax=1, cmap='gray'); ax[0,1].set_title('V', fontsize=14);
    ax[1,0].imshow(imgQueenHSV[:,:,1], vmin=0, vmax=1, cmap='jet'); ax[1,0].set_title('S', fontsize=14);
    ax[1,1].imshow(imgQueenHSV[:,:,0], vmin=0, vmax=1, cmap='jet'); ax[1,1].set_title('H', fontsize=14);
    
    mask = (imgQueenHSV[:,:,0]>0.4) & (imgQueenHSV[:,:,0]<0.6)
    mask1 = mask
    
    mask = mask*1.
    mask_gauss = filters.gaussian(mask, sigma=9, truncate=5)
    
    newH = np.ones(rgbQueen[:,:,0].shape)*imgHSV[0,0,0]
    newH[416:832,200:616] = imgHSV[:,:,0]
    
    newS = np.ones(rgbQueen[:,:,0].shape)*imgHSV[0,0,1]
    newS[416:832,200:616] = imgHSV[:,:,1]
    
    # newH[newH<0.5] = -1
    # newV = np.ones(rgbQueen[:,:,0].shape)*imgHSV[0,0,2]
    # newV[416:832,200:616] = imgHSV[:,:,2]
    
    mask1 = mask_gauss>0.5
    
    outputH = mask1*newH
    outputS = mask1*newS
    # outputV = mask_gauss*newV
    
    imgQueenHSV[mask1,0] = outputH[mask1]
    imgQueenHSV[mask1,1] = outputS[mask1]
    # imgQueenHSV[mask1,2] = outputV[mask1]
    
    imgQueenOutput = color.hsv2rgb(imgQueenHSV)
    
    plt.figure()
    io.imshow(imgQueenOutput)
    return imgQueenOutput

def dressQueenHSVBox(rgbQueen: np.array, rgbImg: np.array) -> np.array:
    imgQueenHSV = color.rgb2hsv(rgbQueen)
    imgHSV = color.rgb2hsv(rgbImg)
    
    fig, ax = plt.subplots(2, 2, figsize=(12,9), dpi=80);
    tight_layout();
    ax[0,0].imshow(rgbQueen); ax[0,0].set_title('RGB', fontsize=14);
    ax[0,1].imshow(imgQueenHSV[:,:,2], vmin=0, vmax=1, cmap='gray'); ax[0,1].set_title('V', fontsize=14);
    ax[1,0].imshow(imgQueenHSV[:,:,1], vmin=0, vmax=1, cmap='jet'); ax[1,0].set_title('S', fontsize=14);
    ax[1,1].imshow(imgQueenHSV[:,:,0], vmin=0, vmax=1, cmap='jet'); ax[1,1].set_title('H', fontsize=14);
    
    mask = (imgQueenHSV[:,:,0]>0.4) & (imgQueenHSV[:,:,0]<0.6)
    mask1 = mask
    
    mask = mask*1.
    box_mask = np.ones([30,30], dtype=float)/900
    box_mask = ndimage.correlate(mask, box_mask)
    
    newH = np.ones(rgbQueen[:,:,0].shape)*imgHSV[0,0,0]
    newH[416:832,200:616] = imgHSV[:,:,0]
    
    newS = np.ones(rgbQueen[:,:,0].shape)*imgHSV[0,0,1]
    newS[416:832,200:616] = imgHSV[:,:,1]
    
    mask1 = box_mask>0.5
    
    imgQueenHSV[:,:,0] = mask1*newH + ~mask1*imgQueenHSV[:,:,0]
    imgQueenHSV[:,:,1] = mask1*newS + ~mask1*imgQueenHSV[:,:,1]
    
    imgQueenOutput = color.hsv2rgb(imgQueenHSV)
    
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