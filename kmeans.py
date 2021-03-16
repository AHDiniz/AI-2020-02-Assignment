#!/usr/bin/python3

import time
import numpy as np
import clustering as clt

def kmeans(clusters : clt.Clusters) -> (clt.Clusters, float):
    
    # Iterating until clock hits one second:
    init_time : float = time.time()
    elapsed_time : float = 0

    prev_sse : float = clusters.sse
    current_sse : float = 0

    while True:
        # Getting the closest centroid for each point and putting it to the cluster it belongs:
        for i in range(0, len(clusters.clusters) - 1):
            point : np.array = clusters.points[i]
            closest : int = -1
            closest_dist : float = np.inf
            for j in range(0, clusters.k - 1):
                d : float = clt.euclidian_dist(point, clusters.centroids[j])
                if d < closest_dist:
                    closest_dist = d
                    closest = j
            clusters.move_point(i, closest)
            
        # Recalculating the clusters for the new configuration:
        for j in range(0, clusters.k - 1):
            clusters.centroid[j] = clusters.cluster_centroid(j)
        
        # Checking if the algorithm still has time to execute.
        # Also stops if the variation of the sse is small enough:
        elapsed_time = time.time()
        prev_sse = current_sse if current_sse != 0 else prev_sse
        current_sse = clusters.sse
        if elapsed_time - init_temp >= 1 or (current_sse - prev_sse) / prev_sse <= .01:
            break

    return (clusters, elapsed_time)
