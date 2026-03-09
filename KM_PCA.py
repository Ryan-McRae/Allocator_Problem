import math
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import panel as pn

#___________________    CONFIGURATION       ___________________

labels = np.array(["110", "111", "112", "113", "114", "115"])
loc_df = pd.read_csv('locationData.csv')
X = loc_df.loc[:, "longitude": "latitude"].to_numpy()                               # (N,2) array
n_points = len(X)
k = math.ceil(n_points / len(labels))                                               # Number of clusters
# PCA
pca = PCA(n_components=2)
pca.fit(X)
principal_direction = pca.components_[0]

#___________________    HAVERSINES       ___________________

def haversine_toCentroid(centroids, points):                                        # Haversine -> point to centroid 
    R = 6371000                                                                     # Earth radius in metres
    points_array = np.radians(points[:, np.newaxis, :])                             # Size -> (N, 1, 2)
    centroids_array = np.radians(centroids[np.newaxis, :, :])                       # Size -> (1, M, 2)
    delta_long = centroids_array[:, :, 0] - points_array[:, :, 0]                   # Size -> (N, M)
    delta_lat = centroids_array[:, :, 1] - points_array[:, :, 1]                    # Size -> (N, M)
    
    a = np.sin(delta_lat/2)**2 + \
        np.cos(points_array[:, :, 1]) * np.cos(centroids_array[:, :, 1]) * np.sin(delta_long/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c                                                                    # Returns -> (N, M)
    
def haversin_points(points):                                                        # Haversine -> point to point
    R = 6371000                                                                     # Earth radius in metres
    points_array = np.radians(points)
    delta_long = points_array[np.newaxis, :, 0] - points_array[:, np.newaxis, 0]
    delta_lat = points_array[np.newaxis, :, 1] - points_array[:, np.newaxis, 1]

    a = np.sin(delta_lat/2)**2 + \
    np.cos(points_array[np.newaxis, :, 1]) * np.cos(points_array[:, np.newaxis, 1]) * np.sin(delta_long/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c                                                                    # Size -> (N, N)

#___________________    Allo Function       ___________________

def allocator(final_seed=None):
    if final_seed is None:
        initial_assignments = np.arange(n_points)
        assignments = initial_assignments % k
        np.random.shuffle(assignments)
    else: 
        assignments = final_seed
    seed = assignments.copy()
    new_assignments = np.zeros(n_points)
    
    while True:
        cluster_centroids = np.zeros((k, 2))
        for Ci in range(k):
            Ci_idx = np.where(assignments == Ci)[0]
            if len(Ci_idx) > 0:
                cluster_centroids[Ci] = np.mean(X[Ci_idx], axis=0)
            else:
                cluster_centroids[Ci] = X[np.random.choice(len(X))]
        PtC_distances = haversine_toCentroid(cluster_centroids, X)
        new_assignments = np.argmin(PtC_distances, axis=1)
        
        if np.array_equal(new_assignments, assignments):
            break
        else:
            assignments = new_assignments
    
    projections = (X - pca.mean_) @ principal_direction
    final_labels = np.empty(n_points, dtype=object)
    for Ci in range(k):
        assigned_idx = np.where(assignments == Ci)[0]
        Ci_projected = projections[assigned_idx]
        Ci_sorted = np.argsort(Ci_projected) % len(labels)
        final_labels[assigned_idx] = labels[Ci_sorted]
    
    scores = np.zeros(len(labels))
    for Li, lbl in enumerate(labels):
        lbl_indices = np.where(final_labels == lbl)[0]
        if len(lbl_indices) > 1:
            point_distances = haversin_points(X[lbl_indices])
            lbl_distances = point_distances[np.triu_indices_from(point_distances, k=1)]
            scores[Li] = np.min(lbl_distances)
        else:
            scores[Li] = 0
    valid_scores = scores[scores > 0]

    if valid_scores.size > 0:
        min_score = np.min(valid_scores)
        avg_score = np.mean(valid_scores)
    else:
        min_score = 0
        avg_score = 0
    score = min_score + avg_score
    
    return min_score, avg_score, score, seed, cluster_centroids, final_labels, new_assignments


def main():
    n = 10000
    top_score = 0
    top_seed = None
    start = time.perf_counter()
    while n > 0:
        min_score, avg_score, score, seed, cluster_centroids, final_labels, new_assignments = allocator()
        if score > top_score:
            top_score = score
            top_seed = seed
        n -=1
    stop = time.perf_counter()
    print(f"SEED : {top_seed} \n SCORE : {top_score} \n TIME : {stop - start:0.4f}s" )

if __name__ == '__main__':
    main()