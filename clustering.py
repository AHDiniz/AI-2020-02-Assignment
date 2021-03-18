#!/usr/bin/python3

import math
import random
import time
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

        # Setting up the lists of points index set:
        self._clusters : np.array = np.zeros(self._num_points)
        points_per_cluster : int = self._num_points / k
        for i in range(0, k):
            np.put(self._clusters, range(int(i * points_per_cluster), int(i * points_per_cluster + points_per_cluster - 1)), i)
        
        # Calculating the initial values of the centroids:
        self._centroids = list([])
        for i in range(0, k - 1):
            self._centroids.append(self.cluster_centroid(i))
    
    # Getting the number of clusters:
    @property
    def k(self) -> int:
        return self._k
    
    # Returning the cluster identifier of each point:
    @property
    def clusters(self) -> np.array:
        return self._clusters

    @property
    def points(self) -> np.array:
        return self._points_data
    
    # Returning the list of centroids of each cluster:
    @property
    def centroids(self) -> list:
        return self._centroids
    
    # Setting the value of the clusters array:
    def clusters(self, clusters : np.array):
        self._clusters = np.copy(clusters)
    
    # Setting the value of the centroids list:
    def centroids(self, centroids : list):
        self._centroids = centroids.copy()
    
    # Getting a single point:
    def get_point(self, point_index : int = 0) -> np.array:
        return self._points_data[point_index]
    
    # Getting array of points that are in a given cluster:
    def get_points_in_cluster(self, cluster : int = 0) -> np.array:
        result = []
        indices = []

        for i in range(0, np.size(self._clusters)):
            if self._clusters[i] == cluster:
                indices.append(i)

        for i in range(int(self._num_points)):
            if i in indices:
                result.append(self._points_data[i])

        return np.array(result)
    
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
            #print(point)
        
        return centroid
    
    # Calcuting the sum of squares of euclidian distances:
    @property
    def sse(self) -> float:
        result = 0
        for c in range(0, self._k - 1):
            points : np.array = self.get_points_in_cluster(c)
            centroid : np.array = self.cluster_centroid(c)
            sse = 0
            for point in points:
                sse += euclidian_dist(point, centroid)
            result += sse
        return result

    # Perturbating a given cluster's centroid:
    def disturb_cluster_centroid(self, cluster_id : int = 0) -> np.array:
        centroid : np.array = np.copy(self._centroids[cluster_id])
        
        random.seed(time.thread_time_ns())

        delta : float = random.random() / 10
        G : float = random.gauss(0, np.linalg.norm(centroid))

        for a in centroid:
            a += delta * G

        return centroid

    # Disturbing the current state to create a possible neighbour:
    def disturb(self):
        # Disturbance will be achieved by perturbating the centroid of a random chosen cluster
        # And then rearraging the points to the cluster with the closest centroid

        disturbed = Clusters(self._k, self._points_data, self._point_dim)
        disturbed.clusters = self._clusters
        disturbed.centroids = self._centroids

        centroid_id : int = random.randint(0, self._k - 1)
        disturbed.centroids[centroid_id] : np.array = np.copy(disturbed.disturb_cluster_centroid(centroid_id))

        for i in range(0, np.ma.size(disturbed.clusters) - 1):
            point : np.array = disturbed.points[i]
            closest : int = 0
            dist_closest : float = math.inf
            for j in range(0, len(disturbed.centroids) - 1):
                d : float = euclidian_dist(point, disturbed.centroids[i])
                if (d < dist_closest):
                    dist_closest = d
                    closest = j
            disturbed.move_point(i, closest)
        
        return disturbed
