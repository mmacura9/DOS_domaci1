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
from pylab import *
from pylab import *
import imageio
from scipy import ndimage

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import numpy as np

def dressQueenHSVGauss(rgbQueen: np.array, yuvImg: np.array) -> np.array:
    imgQueenHSV = color.rgb2hsv(rgbQueen)
    # imgHSV = color.rgb2hsv(rgbImg)
    imgQueenYUV = color.rgb2yuv(rgbQueen)
    yComp = imgQueenYUV[:,:,0]/np.max(imgQueenYUV[:,:,0])
    size1 = rgbQueen.shape
    
    # fig, ax = plt.subplots(2, 2, figsize=(12,9), dpi=80);
    # tight_layout();
    # ax[0,0].imshow(rgbQueen); ax[0,0].set_title('RGB', fontsize=14);
    # ax[0,1].imshow(imgQueenHSV[:,:,2], vmin=0, vmax=1, cmap='gray'); ax[0,1].set_title('V', fontsize=14);
    # ax[1,0].imshow(imgQueenHSV[:,:,1], vmin=0, vmax=1, cmap='jet'); ax[1,0].set_title('S', fontsize=14);
    # ax[1,1].imshow(imgQueenHSV[:,:,0], vmin=0, vmax=1, cmap='jet'); ax[1,1].set_title('H', fontsize=14);
    
    mask = (imgQueenHSV[:,:,0]>0.2) & (imgQueenHSV[:,:,0]<0.5) & (imgQueenHSV[:,:,1]>0.7)
    
    mask = mask*imgQueenHSV[:,:,1]
    mask_gauss = filters.gaussian(mask, sigma=10, truncate=0.1)
    
    newY = imgYUV[0:size1[0],0:size1[1],0]
    
    newU = imgYUV[0:size1[0],0:size1[1],1]
    
    newV = imgYUV[0:size1[0],0:size1[1],2]
    
    mask1 = mask_gauss>0.4
    
    if np.sum(mask1)<6000:
        return rgbQueen
    
    # print(np.sum(mask1))
    outputU = mask1*newU
    outputV = mask1*newV
    
    imgQueenYUV[mask1,0] = newY[mask1]*yComp[mask1]
    imgQueenYUV[mask1,1] = outputU[mask1]
    imgQueenYUV[mask1,2] = outputV[mask1]
    
    imgQueenOutput = color.yuv2rgb(imgQueenYUV)
    
    return imgQueenOutput


if __name__=="__main__":
    imgQueen = imread('queen_dress.jpg')
    rgbImg = imread('slika.jpg')
    imgYUV = color.rgb2yuv(rgbImg)
    filename = 'queen_coat.mp4'
    
    vid = imageio.get_reader(filename,  'ffmpeg')
    video_out = imageio.get_writer('queen_coat_out.mp4', format='FFMPEG', mode = 'I', fps = 30, codec = 'h264')
    for cur_frame in vid:
        cur_frame = dressQueenHSVGauss(cur_frame, imgYUV)
        video_out.append_data(cur_frame)
    
    video_out.close()
    # dressQueenLCH(imgQueen, imgRacism)
    
    # dressQueenHSVGauss(imgQueen, imgRacism)
    
    # dressQueenHSVBox(imgQueen, imgRacism)