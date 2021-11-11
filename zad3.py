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
from skimage import *
import os

def makeHist2D(imgIn: np.array) -> np.array:
    """

    Parameters
    ----------
    imgIn : np.array
        Input image- gray scale.

    Returns
    -------
    np.array
        Histogram of the gray scale image.

    """
    numHist = np.zeros(256, dtype = int)
    for i in range(0, 256):
        numHist[i] = np.sum(imgIn == i)
    return numHist/imgIn.size

def makeHist3D(imgIn: np.array) -> np.array:
    """

    Parameters
    ----------
    imgIn : np.array
        Input image- rgb.

    Returns
    -------
    np.array
        Return 3 histograms for each component of rgb image.

    """
    numHist = np.zeros([256,3], dtype = int)
    for i in range(0, 256):
        numHist[i,0]=np.sum(imgIn[:,:,0]==i)
        numHist[i,1]=np.sum(imgIn[:,:,1]==i)
        numHist[i,2]=np.sum(imgIn[:,:,2]==i)
    return numHist/imgIn[:,:,0].size

def makeMat3D(imgIn: np.array, sizeX: int, sizeY: int, i: int, j: int) -> np.array:
    """
    

    Parameters
    ----------
    imgIn : np.array
        Input rgb image.
    sizeX : int
        Size of x coordinate of the tile.
    sizeY : int
        Size of y coordinate of the tile.
    i : int
        index.
    j : int
        index.
    Returns
    -------
    arr : np.array
        Returns the (i, j) piece of input image

    """
    if (i+1)*sizeX <= imgIn.shape[0] and (j+1)*sizeY <= imgIn.shape[1]:
        return imgIn[i*sizeX:(i+1)*sizeX, j*sizeY:(j+1)*sizeY, :]
    if (i+1)*sizeX > imgIn.shape[0] and (j+1)*sizeY <= imgIn.shape[1]:
        mat = imgIn[i*sizeX:, j*sizeY:(j+1)*sizeY, :]
        con = imgIn[imgIn.shape[0]-(i+1)*sizeX:, j*sizeY:(j+1)*sizeY, :]
        return np.concatenate((mat, con), axis = 0)
    if (i+1)*sizeX <= imgIn.shape[0] and (j+1)*sizeY > imgIn.shape[1]:
        mat = imgIn[i*sizeX:(i+1)*sizeX, j*sizeY:, :]
        con = imgIn[i*sizeX:(i+1)*sizeX, imgIn.shape[1]-(j+1)*sizeY:, :]
        return np.concatenate((mat, con), axis = 1)
    mat = imgIn[i*sizeX:, j*sizeY:, :]
    con1 = imgIn[i*sizeX:, imgIn.shape[1]-(j+1)*sizeY:, :]
    mat = np.concatenate((mat, con1), axis = 1)
    con2 = mat[imgIn.shape[0]-(i+1)*sizeX:, :, :]
    return np.concatenate((mat, con2), axis = 0)

def makeMat2D(imgIn: np.array, sizeX: int, sizeY: int, i: int, j: int) -> np.array:
    """
    

    Parameters
    ----------
    imgIn : np.array
        Input gray-scale image.
    sizeX : int
        Size of x coordinate of the tile.
    sizeY : int
        Size of y coordinate of the tile.
    i : int
        index.
    j : int
        index.
    Returns
    -------
    arr : np.array
        Returns the (i, j) piece of input image

    """
    if (i+1)*sizeX <= imgIn.shape[0] and (j+1)*sizeY <= imgIn.shape[1]:
        return imgIn[i*sizeX:(i+1)*sizeX, j*sizeY:(j+1)*sizeY]
    if (i+1)*sizeX > imgIn.shape[0] and (j+1)*sizeY <= imgIn.shape[1]:
        mat = imgIn[i*sizeX:, j*sizeY:(j+1)*sizeY]
        con = imgIn[imgIn.shape[0]-(i+1)*sizeX:, j*sizeY:(j+1)*sizeY]
        return np.concatenate((mat, con), axis = 0)
    if (i+1)*sizeX <= imgIn.shape[0] and (j+1)*sizeY > imgIn.shape[1]:
        mat = imgIn[i*sizeX:(i+1)*sizeX, j*sizeY:]
        con = imgIn[i*sizeX:(i+1)*sizeX, imgIn.shape[1]-(j+1)*sizeY:]
        return np.concatenate((mat, con), axis = 1)
    mat = imgIn[i*sizeX:, j*sizeY:]
    con1 = imgIn[i*sizeX:, imgIn.shape[1]-(j+1)*sizeY:]
    mat = np.concatenate((mat, con1), axis = 1)
    con2 = mat[imgIn.shape[0]-(i+1)*sizeX:, :]
    return np.concatenate((mat, con2), axis = 0)

def makeT(histogram: np.array, limit: float) -> np.array:
    """
    

    Parameters
    ----------
    histogram : np.array
        Values of histogram.
    limit : float
        Limit for histogram.

    Returns
    -------
    T : np.array
        Returns the distribution function and limits the histogram.

    """
    T1 = sum(histogram[histogram>limit])-sum(histogram>limit)*limit
    histogram[histogram>limit] = limit
    histogram = histogram + T1/255
    T = np.cumsum(histogram)
    T = T*255
    return T

def sgn(x: int) -> int:
    if (x<0):
        return -1
    return 1

def bilinearInterpolation3D(imgIn: np.array, T: np.array, sizeX: int, sizeY: int, numTiles: []) -> np.array:
    """
    

    Parameters
    ----------
    imgIn : np.array
        Input rgb image.
    T : np.array
        Matrix of distribution functions for each tile.
    sizeX : int
        Number of pixcels in x coordinate for each tile.
    sizeY : int
        Number of pixcels in y coordinate for each tile.
    numTiles : []
        Number of tiles.

    Returns
    -------
    imgOut : np.array
        Calculates values of each pixcel using bilinear interpolation.

    """
    imgOut = np.zeros(imgIn.shape, dtype = uint8)
    for i in range(imgIn.shape[0]):
        for j in range(imgIn.shape[1]):
            blockX = math.floor(i/sizeX)
            blockY = math.floor(j/sizeY)
            
            centerX = sizeX/2 + blockX*sizeX
            centerY = sizeY/2 + blockY*sizeY
            pomX = sgn(i-centerX)
            pomY = sgn(j-centerY)
            
            a = abs(i-centerX)
            b = abs(i-(centerX+pomX*sizeX))
            
            x1 = np.array([0, 0, 0])
            x2 = np.array([0, 0, 0])
            x3 = np.array([0, 0, 0])
            
            if blockX+pomX<0 or (blockX+pomX)>=numTiles[0]:
                a = 0
            else:
                x10 = T[blockX+pomX,blockY,imgIn[i,j,0],0]
                x11 = T[blockX+pomX,blockY,imgIn[i,j,1],1]
                x12 = T[blockX+pomX,blockY,imgIn[i,j,2],2]
                x1 = np.array([x10, x11, x12])
            
            if blockY+pomY>=0 and (blockY+pomY)<numTiles[1]:
                x20 = T[blockX,blockY+pomY,imgIn[i,j,0],0]
                x21 = T[blockX,blockY+pomY,imgIn[i,j,1],1]
                x22 = T[blockX,blockY+pomY,imgIn[i,j,2],2]
                x2 = np.array([x20, x21, x22])
                if blockX+pomX>=0 and (blockX+pomX)<numTiles[0]:
                    x30 = T[blockX+pomX,blockY+pomY,imgIn[i,j,0],0]
                    x31 = T[blockX+pomX,blockY+pomY,imgIn[i,j,1],1]
                    x32 = T[blockX+pomX,blockY+pomY,imgIn[i,j,2],2]
                    x3 = np.array([x30, x31, x32])
            
            c = abs(j-centerY)
            d = abs(j-(centerY+pomY*sizeY))
            
            if blockY+pomY<0 or (blockY+pomY)>=numTiles[1]:
                c = 0
            
            sh10 = (T[blockX,blockY,imgIn[i,j,0],0]*b + x1[0]*a)/(a+b)
            
            sh11 = (T[blockX,blockY,imgIn[i,j,1],1]*b + x1[1]*a)/(a+b)
            
            sh12 = (T[blockX,blockY,imgIn[i,j,2],2]*b + x1[2]*a)/(a+b)
            
            sh1 = np.array([sh10, sh11, sh12])
            sh2 = (x2*b+x3*a)/(a+b)
            
            imgOut[i,j,:] = np.round((sh1*d+sh2*c)/(c+d))
    
    return imgOut

def bilinearInterpolation2D(imgIn: np.array, T: np.array, sizeX: int, sizeY: int, numTiles: []) -> np.array:
    """
    

    Parameters
    ----------
    imgIn : np.array
        Input gray-scale image.
    T : np.array
        Matrix of distribution functions for each tile.
    sizeX : int
        Number of pixcels in x coordinate for each tile.
    sizeY : int
        Number of pixcels in y coordinate for each tile.
    numTiles : []
        Number of tiles.

    Returns
    -------
    imgOut : np.array
        Calculates values of each pixcel using bilinear interpolation.

    """
    imgOut = np.zeros(imgIn.shape, dtype = uint8)
    for i in range(imgIn.shape[0]):
        for j in range(imgIn.shape[1]):
            blockX = math.floor(i/sizeX)
            blockY = math.floor(j/sizeY)
            
            centerX = sizeX/2 + blockX*sizeX
            centerY = sizeY/2 + blockY*sizeY
            pomX = sgn(i-centerX)
            pomY = sgn(j-centerY)
            
            a = abs(i-centerX)
            b = abs(i-(centerX+pomX*sizeX))
            
            x1 = 0
            x2 = 0
            x3 = 0
            
            if blockX+pomX<0 or (blockX+pomX)>=numTiles[0]:
                a = 0
            else:
                x1 = T[blockX+pomX,blockY,imgIn[i,j]]
            
            if blockY+pomY>=0 and (blockY+pomY)<numTiles[1]:
                x2 = T[blockX,blockY+pomY,imgIn[i,j]]
                if blockX+pomX>=0 and (blockX+pomX)<numTiles[0]:
                    x3 = T[blockX+pomX,blockY+pomY,imgIn[i,j]]
            
            c = abs(j-centerY)
            d = abs(j-(centerY+pomY*sizeY))
            
            if blockY+pomY<0 or (blockY+pomY)>=numTiles[1]:
                c = 0
            
            sh1 = (T[blockX,blockY,imgIn[i,j]]*b + x1*a)/(a+b)
            sh2 = (x2*b+x3*a)/(a+b)
            
            imgOut[i,j] = np.round((sh1*d+sh2*c)/(c+d))
    
    return imgOut

def dosCLAHE(imgIn: np.array, numTiles: [] = [8, 8], limit: float = 0.01) -> np.array:
    """
    

    Parameters
    ----------
    imgIn : np.array
        Input image.
    numTiles : [], optional
        Number of tiles. The default is [8, 8].
    limit : float, optional
        Clip limit for histogram values. The default is 0.01.

    Returns
    -------
    TYPE
        Returns tha image processed by CLAHE algorithm.

    """
    sizeX = math.ceil(imgIn.shape[0]/numTiles[0])
    sizeY = math.ceil(imgIn.shape[1]/numTiles[1])
    if len(imgIn.shape) == 3:
        print("3D Slika")
        T = np.zeros([numTiles[0],numTiles[1],256,3])
        for i in range(numTiles[0]):
            for j in range(numTiles[1]):
                mat = makeMat3D(imgIn, sizeX, sizeY, i, j)
                histogram = makeHist3D(mat)
                T[i, j, :, 0] = makeT(histogram[:, 0], limit)
                T[i, j, :, 1] = makeT(histogram[:, 1], limit)
                T[i, j, :, 2] = makeT(histogram[:, 2], limit)
        return bilinearInterpolation3D(imgIn, T, sizeX, sizeY, numTiles)
    else:
        print("2D Slika")
        T = np.zeros([numTiles[0], numTiles[1], 256])
        for i in range(numTiles[0]):
            for j in range(numTiles[1]):
                mat = makeMat2D(imgIn, sizeX, sizeY, i, j)
                histogram = makeHist2D(mat)
                T[i,j,:] = makeT(histogram, limit)
        return bilinearInterpolation2D(imgIn, T, sizeX, sizeY, numTiles)
                
if __name__ == "__main__":
    imgIn = imread('train.jpg')
    img1 = color.rgb2yuv(imgIn)
    plt.figure()
    io.imshow(imgIn)
    plt.figure()
    io.imshow(dosCLAHE(imgIn,[16, 16], 0.01))
    # plt.figure()
    # io.imshow(dosCLAHE(img_as_ubyte(img1[:,:,0])))

    