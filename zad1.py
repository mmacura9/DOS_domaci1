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
import imageio
from scipy import ndimage

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import numpy as np

def dressQueen(rgbQueen: np.array, yuvImg: np.array) -> np.array:
    imgQueenHSV = color.rgb2hsv(rgbQueen)
    # imgHSV = color.rgb2hsv(rgbImg)
    mask = (imgQueenHSV[:,:,0]>0.25) & (imgQueenHSV[:,:,0]<0.5) & (imgQueenHSV[:,:,1]>=0.4*np.max(imgQueenHSV[:,:,1]))
    
    mask = mask*1.#imgQueenHSV[:,:,1]
    mask_gauss = filters.gaussian(mask, sigma=20, truncate=0.25)
    
    mask1 = mask_gauss>0.7
    print(np.sum(mask1))
    if np.sum(mask1)<10000:
        return rgbQueen
    
    imgQueenYUV = color.rgb2yuv(rgbQueen)
    yComp = imgQueenYUV[:,:,0]/np.max(imgQueenYUV[:,:,0])
    size1 = rgbQueen.shape
    
    newY = imgYUV[0:size1[0],0:size1[1],0]
    
    newU = imgYUV[0:size1[0],0:size1[1],1]
    
    newV = imgYUV[0:size1[0],0:size1[1],2]
    
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
    rgbImg = imread('jazavicar1.jpg')
    
    imgYUV = color.rgb2yuv(rgbImg)
    # filename = 'queen_coat.mp4'
    
    # vid = imageio.get_reader(filename,  'ffmpeg')
    # video_out = imageio.get_writer('queen_coat_out.mp4', format='FFMPEG', mode = 'I', fps = 30, codec = 'h264')
    # i=0
    # for cur_frame in vid:
    #     i=i+1
    #     if i>900:
    #         cur_frame = dressQueen(cur_frame, imgYUV)
    #         video_out.append_data(cur_frame)
        
    # video_out.close()
    
    imshow(dressQueen(imgQueen, imgYUV))
