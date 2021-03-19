#!/usr/bin/python3

import math
import random
import time
import copy
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
        self._num_points : int = self._points_data.shape[0]

        # Creating a random seed:
        random.seed(time.thread_time_ns())

        # Initializing the data arrays:
        self._clusters : np.array = np.zeros(self._num_points)
        self._centroids = list([])
        for i in range(0, self._k):
            self._centroids.append(np.zeros(point_dim))
    
    # Initializing the state of the clusters:
    def initialize_state(self):
        # Setting up the lists of points index set:
        for i in range(self._num_points):
            self._clusters[i] = random.randint(0, self._k - 1)
        
        # Calculating the initial values of the centroids:
        for i in range(self._k):
            self._centroids[i] = self.cluster_centroid(i)

    # Getting the number of clusters:
    def get_k(self) -> int:
        return self._k
    
    # Returning the cluster identifier of each point:
    def get_clusters(self) -> np.array:
        return self._clusters

    # Returning the points data in the clusters:
    def get_points(self) -> np.array:
        return self._points_data
    
    # Returning the list of centroids of each cluster:
    def get_centroids(self) -> list:
        return self._centroids
    
    # Returning the number of points in the set:
    def get_num_points(self) -> int:
        return self._num_points
    
    # Setting the value of the clusters array:
    def set_clusters(self, clusters : np.array):
        self._clusters = np.copy(clusters)
    
    # Setting the value of the centroids list:
    def set_centroids(self, centroids : list):
        self._centroids = centroids.copy()
    
    # Getting a single point:
    def get_point(self, point_index : int = 0) -> np.array:
        return self._points_data[point_index]
    
    # Getting array of points that are in a given cluster:
    def get_points_in_cluster(self, cluster : int = 0) -> np.array:
        indices : np.array = np.where(self.clusters == cluster)[0]
        result : np.array = np.take(self.points, indices, axis=0)
        return result
    
    # Changing the cluster of a given point:
    def move_point(self, point_index : int = 0, cluster_id : int = 0):
        self._clusters[point_index] = cluster_id
    
    # Calculating the centroid (mean point) of the desired cluster:
    def cluster_centroid(self, cluster_id : int = 0) -> np.array:
        centroid : np.array = np.zeros(self._point_dim[1]) # Initializing the centroid variable

        # Getting the points that are inside the target cluster:
        points : np.array = self.get_points_in_cluster(cluster_id)
        num_points : int = points.shape[0]

        # Calculating the centroid coordinates:
        for point in points:
            centroid += point / num_points
        
        return centroid
    
    # Calcuting the sum of squares of euclidian distances:
    def get_sse(self) -> float:
        result = 0
        for c in range(0, self._k - 1):
            points : np.array = self.get_points_in_cluster(c)
            centroid : np.array = self.cluster_centroid(c)
            sse = 0
            for point in points:
                sse += euclidian_dist(point, centroid) ** 2
            result += sse
        return result

    # Getting the point in point_set that is furthest from c
    def furthest_in_set(self, point_set : np.array, c : np.array) -> np.array:
        f = lambda p : euclidian_dist(p, c)
        return (np.apply_along_axis(f, 1, point_set)).argmax()

    # Disturbing the current state to create a possible neighbour:
    def disturb(self):
        # Disturbance will be achieved by selecting a random cluster and removing the furthest point to the cluster with closest centroid

        disturbed = Clusters(self._k, self._points_data, self._point_dim[1])
        disturbed.clusters = copy.deepcopy(self._clusters)
        disturbed.centroids = copy.deepcopy(self._centroids)

        disturbed.clusters[random.randint(0, disturbed.num_points - 1)] = random.randint(0, disturbed.k - 1)

        return disturbed

    k = property(get_k)
    points = property(get_points)
    num_points = property(get_num_points)
    sse = property(get_sse)
    clusters = property(get_clusters, set_clusters)
    centroids = property(get_centroids, set_centroids)