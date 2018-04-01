# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 23:47:43 2018

@author: Kruthi
"""

from skimage import io,novice
import numpy as np

import k_means_appln as kmeans

picture = novice.open('test.jpg')
picture.show()

image = io.imread('test.jpg')
print(image.shape)

rows = image.shape[0]
cols = image.shape[1]
image = image.reshape(rows*cols,3)
print(image.shape)

clusters = np.zeros(len(image),dtype=int)
print(clusters.shape)
print(clusters)

centroids = kmeans.randomInit(image,16,len(image),3)
print(centroids)

(finalCentroids,finalClusters) = kmeans.kmeans_call(image,clusters,centroids,3)

for i in range(len(image)):
    image[i] = finalCentroids[finalClusters[i]]
    
converted = np.asarray(image) 
converted = converted.reshape(rows,cols,3)
io.imsave('converted.jpg',converted)
comppicture = novice.open('converted.jpg')
comppicture.show()