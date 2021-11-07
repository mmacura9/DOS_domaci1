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
    numHist = np.zeros(255, dtype = int)
    for i in range(0, 255):
        numHist[i] = np.sum(imgIn == i)
    return numHist/imgIn.size

def makeHist3D(imgIn: np.array) -> np.array:
    numHist = np.zeros([255,3], dtype = int)
    for i in range(0, 255):
        numHist[i,0]=np.sum(imgIn[:,:,0]==i)
        numHist[i,1]=np.sum(imgIn[:,:,1]==i)
        numHist[i,2]=np.sum(imgIn[:,:,2]==i)
    return numHist/imgIn.size

def makeMat3D(imgIn: np.array, sizeX: int, sizeY: int, i: int, j: int) -> np.array:
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
    T = np.cumsum(histogram)
    T[T>limit] = limit
    T = T+limit
    T = T*255
    return T

def dosCLAHE(imgIn: np.array, numTiles: [] = [8, 8], limit: float = 0.01) -> np.array:
    x = np.linspace(0,255,255)
    sizeX = math.ceil(imgIn.shape[0]/numTiles[0])
    sizeY = math.ceil(imgIn.shape[1]/numTiles[1])
    if imgIn.shape[2] == 2:
        print("2D Slika")
        T = np.zeros([numTiles[0], numTiles[1], 255])
        histogram = np.zeros(255)
        for i in range(numTiles[0]):
            for j in range(numTiles[1]):
                mat = makeMat2D(imgIn, sizeX, sizeY, i, j)
                histogram = makeHist2D(mat)
                T[i,j,:] = makeT(histogram, limit)
    else:
        print("3D Slika")
        T = np.zeros([numTiles[0],numTiles[1],255,3])
        histogram = np.zeros([255, 3])
        for i in range(numTiles[0]):
            for j in range(numTiles[1]):
                mat = makeMat3D(imgIn, sizeX, sizeY, i, j)
                histogram = makeHist3D(mat)
                T[i, j, :, 0] = makeT(histogram[:, 0], limit)
                T[i, j, :, 1] = makeT(histogram[:, 1], limit)
                T[i, j, :, 2] = makeT(histogram[:, 2], limit)
        

imgIn = imread('train.jpg')
plt.figure()
io.imshow(imgIn)
dosCLAHE(imgIn,[20,20])