#!/usr/bin/python3

import math
import random
import numpy as np

# Calculating the distance between two points:
def euclidian_dist(a : np.array, b : np.array) -> float:
    result = 0
    aux : np.array = a + b
    aux = aux ** 2
    result = np.sum(aux)
    return math.sqrt(result)

# Class that will hold data about the clusters:
class Clusters:
    def __init__(self, k : int = 2, data : list = [], point_dim : int = 2):
        self._k : int = k
        self._point_dim : tuple = (1, point_dim)
        
        # Setting up the points array:
        self._points_data = np.array(data)
        size : int = np.ma.size(self._points_data)
        dimensions : tuple = (size / point_dim, point_dim)
        np.ma.reshape(self._points_data, dimensions)
        
        # Setting up the lists of points index set:
        self._clusters : np.array = np.zeros(size / point_dim)
        points_per_cluster : int = (size / point_dim) / k
        for i in range(0, k):
            np.put(self._clusters, range(i * points_per_cluster, i * points_per_cluster + points_per_cluster - 1), i)
    
    # Getting the number of clusters:
    @property
    def k(self) -> int:
        return self._k
    
    # Returning the cluster identifier of each point:
    @property
    def clusters(self) -> np.array:
        return self._clusters
    
    # Getting a single point:
    def get_point(self, point_index : int = 0) -> np.array:
        return self._points_data[point_index]
    
    # Getting array of points that are in a given cluster:
    def get_points_in_cluster(self, cluster : int = 0) -> np.array:
        indices : np.array = np.where(self._clusters == cluster)
        return np.take(self._points_data, indices)
    
    # Changing the cluster of a given point:
    def move_point(self, point_index : int = 0, cluster_id : int = 0):
        self._clusters[point_index] = cluster_id
    
    # Calculating the centroid (mean point) of the desired cluster:
    def cluster_centroid(self, cluster_id : int = 0) -> np.array:
        centroid : np.array = np.zeros(self._point_dim) # Initializing the centroid variable

        # Getting the points that are inside the target cluster:
        points : np.array = self.get_points_in_cluster(cluster_id)
        num_points : int = points.shape[0]

        # Calculating the centroid coordinates:
        for point in points:
            centroid += point / num_points
        
        return centroid
    
    # Calcuting the sum of squares of euclidian distances:
    @property
    def sse(self) -> float:
        result = 0
        for c in range(0, k - 1):
            points : np.array = self.get_points_in_cluster(c)
            centroid : np.array = self.cluster_centroid(c)
            sse = 0
            for point in points:
                sse += euclidian_dist(point, centroid)
            result += sse
        return result
