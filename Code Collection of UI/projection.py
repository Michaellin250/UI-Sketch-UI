import torchvision
import numpy as np
import torch
import math
import sys, os

def projectXY(grid, imgWidth, imgHeight):
    binaryRep = convertToBinaryGrid(grid)
    img1 = torch.zeros([imgWidth, imgHeight], dtype=torch.int32)
    Mvp = constructMvp(imgWidth, imgHeight)
    Morth = constructOrth(-2, -2, -2, 2, 2, 2)
    M = np.dot(Mvp, Morth) 
    for i in range(64):
        for j in range(64):
            for k in range(64):
                if binaryRep[i,j,k]:
                    coords = np.reshape([i, j, k, 1], (4, 1))
                    #print(M)
                    #print(coords)
                    p = np.dot(M.astype(int), coords)
                    print(p)
                    #xPixel = p[0,0]
                    #yPixel = p[1, 0]
                    #img1[xPixel, yPixel] = 1
    torchvision.utils.save_image(img1, "./comb/" + "maskZ.png")

def constructMvp(imgWidth, imgHeight):
    Mvp = np.zeros((4,4))
    Mvp[0,0] = imgWidth / 2
    Mvp[0,3] = (imgWidth - 1) / 2
    Mvp[1,1] = imgHeight / 2
    Mvp[1,3] = (imgHeight - 1) / 2
    Mvp[2,2] = 1
    Mvp[3,3] = 1
    return Mvp

def constructOrth(l, b, n, r, t, f):
    Morth = np.zeros((4,4))
    Morth[0,0] = 2 / (r - l)
    Morth[0,3] = -((r + l) / (r - l))
    Morth[1,1] = 2 /(t - b)
    Morth[1,3] = -((t + b) / (t - b))
    Morth[2,2] = 2 / (n - f)
    Morth[2,3] = -((n + f) / (n - f))
    Morth[3,3] = 1
    return Morth

def convertToBinaryGrid(grid):
    binaryGrid = grid.clone()
    binaryGrid[binaryGrid > 0] = 1
    binaryGrid[binaryGrid < 0] = 0
    #print("Binary Grid is")
    #print(binaryGrid)
    return binaryGrid


def printGrid(grid):
    print("Printing every voxel")
    for i in range(64):
        for j in range(64):
            for k in range(64):
                print(grid[i,j,k])

