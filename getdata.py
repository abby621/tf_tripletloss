# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:49:13 2016

@author: souvenir
"""

import numpy as np
import cv2

class TripletSetData:
    def __init__(self, image_list, image_size, crop_size, isTraining=True, shuffle=False):
        self.meanFile = './models/places365/places365CNN_mean.npy'
        tmp = np.load(self.meanFile)
        self.meanImage = np.moveaxis(tmp, 0, -1)

        self.files = []
        if isinstance(image_list,basestring):
            # Reads a .txt file containing image paths of image sets where each line contains
            # all images from the same set and the first image is the anchor
            f = open(image_list, 'r')
            for line in f:
                temp = line[:-1].split(' ')
                self.files.append(temp)
        elif isinstance(image_list,list):
            for line in image_list:
                thisList = []
                for im in line:
                    for elem in im:
                        if '.jpg' in elem.lower():
                            thisList.append(elem)
                self.files.append(thisList)

        self.image_size = image_size
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.isTraining = isTraining
        self.indexes = np.arange(0, len(self.files))
        self.pointer = 0
        if (shuffle):
            self.shuffleIndexes()

    def shuffleIndexes(self):
        self.indexes = np.random.permutation(self.indexes)

    def getBatch(self):
        nPos = len(self.files[self.indexes[self.pointer]])
        nNeg = len(self.files[self.indexes[self.pointer + 1]])

        batch = np.zeros([nPos + nNeg, self.crop_size[0], self.crop_size[1], 3])
        for i in np.arange(nPos):
            batch[i,:,:,:] = self.getProcessedImage(self.files[self.indexes[self.pointer]][i])
        for j in np.arange(nNeg):
            batch[nPos + j,:,:,:] = self.getProcessedImage(self.files[self.indexes[self.pointer + 1]][j])
        self.pointer += 2
        if (self.pointer >= len(self.files) - 1):
            self.pointer = 0;
            if (self.shuffle):
                self.shuffleIndexes()
        return batch, nPos

    def getProcessedImage(self, image_file):
        img = cv2.imread(image_file)
        img = cv2.resize(img, (self.image_size[0], self.image_size[1]))
        img = img - self.meanImage

        if (self.isTraining):
            top = np.random.randint(self.image_size[0] - self.crop_size[0])
            left = np.random.randint(self.image_size[1] - self.crop_size[1])
            img = img[top:(top+self.crop_size[0]),left:(left+self.crop_size[1]),:]
        else:
            img = img[14:(self.crop_size[0] + 14), 14:(self.crop_size[1] + 14),:]

        return img
