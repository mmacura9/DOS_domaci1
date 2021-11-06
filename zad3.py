# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 13:58:38 2021

@author: mm180261d
"""
from pylab import *
import numpy as np
from skimage import color
from skimage import io
import matplotlib.pyplot as plt
import math

def makeHist2D(imgIn: np.array) -> np.array:
    print("Pravim histogram")
    numHist = np.zeros(255, dtype = int)
    for i in range(0, 255):
        numHist[i] = np.sum((imgIn <= i) & (imgIn > i-1))
    return numHist

def makeHist3D(imgIn: np.array) -> np.array:
    numHist = np.zeros([255,3], dtype = int)
    # print(numHist.shape)
    for i in range(0, 255):
        numHist[i,0]=np.sum((imgIn[:,:,0]>=i) & (imgIn[:,:,0]<i+1))
        numHist[i,1]=np.sum((imgIn[:,:,1]>=i) & (imgIn[:,:,1]<i+1))
        numHist[i,2]=np.sum((imgIn[:,:,2]>=i) & (imgIn[:,:,2]<i+1))
    return numHist/np.size(imgIn[:,:,0])

def dosCLAHE(imgIn: np.array, numTiles: [] = [8, 8], limit: float = 0.01) -> np.array:
    x = np.linspace(0,255,255)
    if imgIn.shape[2] == 2:
        print("2D Slika")
        histogram = makeHist2D(imgIn)
    else:
        print("3D Slika")
        # print(makeHist3D(imgIn))
        sizeX = math.ceil(imgIn.shape[0]/numTiles[0])
        sizeY = math.ceil(imgIn.shape[1]/numTiles[1])
        histogram = np.zeros([numTiles[0],numTiles[1],255,3])
        for i in range(numTiles[0]-1):
            for j in range(numTiles[1]-1):
                histogram[i,j,:,:] = makeHist3D(imgIn[i*sizeX:(i+1)*sizeX, j*sizeY:(j+1)*sizeY,:])
                plt.figure()
                plt.stem(histogram[i,j,:,0])
        

imgIn = imread('train.jpg')
plt.figure()
io.imshow(imgIn)
dosCLAHE(imgIn,[20,20])