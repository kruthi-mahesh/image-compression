# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 23:44:51 2018

@author: Kruthi
"""

from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np

def computeDistance(x1,x2):
    '''Euclidean distance'''
    return np.sqrt(np.sum(np.power((x1-x2),2)))

def randomInit(X,no_of_clusters,no_of_samples,nof):
    centroids = []
    for i in range(no_of_clusters):
        temp = np.random.randint(0,no_of_samples)
        tempcents = []
        for j in range(nof):
            tempcents.append(X[temp,j])
        centroids.append(tempcents)
    centroids = np.matrix(centroids)
    return centroids

def labelling(x,centroids):
    distances = {}
    for i in range(len(centroids)):
        distances[computeDistance(x,centroids[i,:])] = i
    return distances[min(distances.keys())]

def clusterAssignment(X,clusters,centroids):
    for i in range(len(X)):
        clusters[i] = labelling(X[i,:],centroids)
    return clusters


def updateCentroid(X,clusters,centroids,nof):
    sums = np.matrix(np.zeros(centroids.shape))
    count = np.zeros(len(centroids),dtype=int)
    newCentroids = np.matrix(np.zeros(centroids.shape))
    for i in range(len(X)):
        for j in range(nof):
            sums[clusters[i],j]+=X[i,j]
        count[clusters[i]]+=1
    for i in range(len(centroids)):
        newCentroids[i,:] = sums[i,:]/count[i]

    return newCentroids

def kmeans_call(X,clusters,centroids,nof): 
    count = 0
    while True:
        count += 1
        print('old centroids')
        print(centroids)
        clusters = clusterAssignment(X,clusters,centroids)
        new_centroids = updateCentroid(X,clusters,centroids,nof)
        print('new centroids')
        print(new_centroids)
        dif = np.abs(np.subtract(new_centroids,centroids))
        print('dif')
        print(dif)
        decision = (dif<0.00001).all()
        print(decision)
        
        if(decision):
            print('count of iterations' + str(count))
            return centroids,clusters  
        centroids = new_centroids
        
def plotCentroids(Centroids):
    for i in range(len(Centroids)):
        plt.scatter(Centroids[i,0],Centroids[i,1],c='y')
    

def plotClusters(Centroids,Clusters,X):
    colors = ['r','g','b','cyan']
    plt.figure()
    for i in range(len(Clusters)):
        plt.scatter(X[i,0],X[i,1],c = colors[Clusters[i]])
    plotCentroids(Centroids)
    plt.show()
    
'''number_of_samples = 200  
number_of_clusters = 3
number_of_iterations = 10
number_of_features = 2
centroids = randomInit(X,number_of_clusters,number_of_samples,number_of_features) 
clusters = np.zeros(len(X),dtype=int)
(finalCentroids,finalClusters) = kmeans_call(X,clusters,centroids,number_of_iterations,number_of_features)
plotClusters(finalCentroids,finalClusters,X)'''
