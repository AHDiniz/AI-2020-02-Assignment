#!/usr/bin/python3

import time
import numpy as np
import scipy as sp
import clustering as clt

def kmeans(k : int, points : np.array) -> (float, float):
    centroids, labels = sp.cluster.vq.kmeans2(points, k, minit = "point")
    init_time = time.time()
    sse : float = 0
    for i in range(points.shape[0]):
        ds : float = clt.euclidian_dist(points.flat[i], centroids[labels[i]]) ** 2
        sse += ds
    end_time = time.time()
    return (sse, end_time - init_time)