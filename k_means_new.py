# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 20:24:15 2018

@author: Kruthi
"""

from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np


# In[51]:


X,y = make_blobs(n_samples=20000,centers=3,n_features = 3,cluster_std=0.60,random_state=0)


# In[52]:


print(X.shape)
print(X)


# In[53]:


print(y.shape)
print(y)


# In[54]:


plt.scatter(X[:,0],X[:,1],c='black')


# In[55]:


def computeDistance(x1,x2):
    '''Euclidean distance'''
    return np.sqrt(np.sum(np.power((x1-x2),2)))


# In[56]:


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


# In[57]:


def labelling(x,centroids):
    distances = {}
    for i in range(len(centroids)):
        distances[computeDistance(x,centroids[i,:])] = i
    return distances[min(distances.keys())]


# In[58]:


def clusterAssignment(X,clusters,centroids):
    for i in range(len(X)):
        clusters[i] = labelling(X[i,:],centroids)
    return clusters


# In[59]:


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


# In[60]:


clusters = np.zeros(len(X),dtype=int)
print(clusters.shape)
print(clusters)


# In[61]:


centroids = randomInit(X,3,len(X),3)


# In[62]:


clusters = clusterAssignment(X,clusters,centroids)
print(clusters)


# In[63]:


new_centroids = updateCentroid(X,clusters,centroids,3)


# In[64]:


print(centroids)
print(new_centroids)
dif = np.abs(np.subtract(new_centroids,centroids))
print(dif)
decision = (dif<0.001).all()
print(decision)


# In[65]:


centroids = new_centroids


# In[66]:


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


# In[49]:


(finalCentroids,finalClusters) = kmeans_call(X,clusters,centroids,3)


