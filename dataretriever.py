# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:49:13 2016

@author: souvenir
"""

import numpy as np
import cv2

class DataRetriever:
    def __init__(self, image_list, image_size, crop_size, batch_size, isTraining=True, shuffle=False):
        self.meanFile = './models/places365/places365CNN_mean.npy'
        tmp = np.load(self.meanFile)
        self.meanImage = np.moveaxis(tmp, 0, -1)

        self.files = []
        for line in image_list:
            thisList = []
            for im in line:
                if '.jpg' in im.lower():
                    thisList.append(im)
            self.files.append(thisList)

        self.batch_size = batch_size
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
        batch = np.zeros([self.batch_size, self.crop_size[0], self.crop_size[1], 3])
        for i in np.arange(self.batch_size):
            batch[i,:,:,:] = self.getProcessedImage(self.files[self.indexes[self.pointer]][0])
            self.pointer += 1
            if (self.pointer >= len(self.files) - 1):
                self.pointer = 0;
                if (self.shuffle):
                    self.shuffleIndexes()
        return batch

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
