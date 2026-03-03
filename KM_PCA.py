import random, math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Configuration ──────────────────────────────────────────────
labels = ["110", "111", "112", "113", "114", "115"]
x1 = [
    -0.03098, -0.02554, -0.02448, -0.02415, -0.02277, 
    -0.02204, -0.02201, -0.02185, -0.02234, -0.02206, 
    -0.02052, -0.02025, -0.01921, -0.01725, -0.01561, 
    -0.01273, -0.01272, -0.01216, -0.01078
]
x2 = [
    51.53657, 51.53833, 51.53721, 51.5445, 51.54439, 
    51.54735, 51.54739, 51.54525, 51.53328, 51.53948, 
    51.54653, 51.54472, 51.54262, 51.53934, 51.53862, 
    51.54337, 51.54202, 51.5407, 51.54023
]
X = np.array(list(zip(x1, x2)))
k = math.ceil(len(X) / len(labels))    # Cluster max
clusters = ["c" + str(i) for i in range(1, k+1)]

#── PCA  ───────────────────────────────────────────────────────
pca = PCA(n_components=2)
pca.fit(X)
principal_direction = pca.components_[0] 

# ── Distance function ──────────────────────────────────────────
def haversine(p1, p2):
    R = 6371000  # Earth radius in meters
    lon1, lat1 = np.radians(p1)
    lon2, lat2 = np.radians(p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ── Cluster + Assignment function ──────────────────────────────
def allocator(final_seed = None):
    if final_seed is None:
        # ── Initial balanced random assignment ────────────────────────
        outputs = []
        for i in range(len(x1)):
            cluster_idx = i // len(labels)
            outputs.append(clusters[cluster_idx])
        random.shuffle(outputs)
        seed = outputs[:] # Store initial seed for scoring
    else:
        outputs = final_seed[:]
        seed = final_seed[:]
    new_outputs = [0] * len(x1)

    # ── K-Means loop ───────────────────────────────────────────────
    while True:
        cluster_centroids = [] # Mapped via indexes, ie. [0] => c1
        for cx in range(len(clusters)):
            total_X = [0, 0]
            count = 0
            for j in range(len(outputs)):
                if clusters[cx] == outputs[j]:
                    count += 1
                    total_X[0] += x1[j]
                    total_X[1] += x2[j]
            if count == 0:
                random_idx = random.randint(0, len(x1)-1)
                cluster_centroids.append([x1[random_idx], x2[random_idx]])
            else:
                cluster_centroids.append([n / count for n in total_X])

            # 2. Reassign each point to its nearest centroid — no size constraint
            for i in range(len(x1)):
                best_cluster = None
                best_dist = float('inf')
                for j in range(len(cluster_centroids)):
                    d = haversine(X[i], cluster_centroids[j])
                    if d < best_dist:
                        best_dist = d
                        best_cluster = clusters[j]
                new_outputs[i] = best_cluster

        if new_outputs == outputs:
            break
        else:
            outputs = new_outputs[:]

    # ── Clusters Grouped ───────────────────────────────────────────────
    clusters_sorted = []
    for j in clusters:
        cx = []
        for i in range(len(outputs)):
            if outputs[i] == j:
                cx.append(i)
        clusters_sorted.append(cx)
    # ── Cluster Groups Projected ───────────────────────────────────────────────
    clusters_projected = []
    for cx in clusters_sorted:
        projection = []
        for i in cx:
            projection.append(np.dot(np.array([x1[i], x2[i]]) - pca.mean_, principal_direction))
        clusters_projected.append(projection)
    # ── Points Based on Position  ───────────────────────────────────────────────
    final_labels = [None] * len(x1)
    for i, proj in enumerate(clusters_projected):
        indices = np.argsort(proj)
        for j,v in enumerate(indices):
            final_labels[clusters_sorted[i][v]] = labels[j if j < 6 else j - 6]
            
    # ── Scoring ───────────────────────────────────────────────
    final_labels_sorted = [] # Group points by final label for scoring
    for j in labels:
        l = []
        for i,v in enumerate(final_labels):
            if v == j:
                l.append(i)
        final_labels_sorted.append(l)
        
    scores = []
    for sub in final_labels_sorted: # In each label group, find min inter-label distance
        inter_label_distances = []
        if len(sub) > 1:
            for i in range(len(sub)):
                for j in range(i+1, len(sub)):
                    inter_label_distances.append(haversine(X[sub[i]], X[sub[j]]))
        else:
            inter_label_distances = [0]
        scores.append(np.min(inter_label_distances))
        
    unique_scores = np.unique(scores) # Find minimum non-zero score for final scoring
    if np.min(scores) == 0:
        min_score = unique_scores[1] 
    else:
        min_score = unique_scores[0]
    score_avg = np.average(scores)
    total_score = min_score * score_avg
    
    return min_score, score_avg, total_score, seed, cluster_centroids, final_labels, outputs

def main():
    n = 10000
    top_score = 0
    top_seed = None
    while n > 0:
        min_score, score_avg, total_score, seed, cluster_centroids, final_labels, outputs = allocator()
        #print(f"Seed: {seed} | Min Score: {min_score:.4f} | Avg Score: {score_avg:.4f} | Total Score: {total_score:.4f}")
        if total_score > top_score:
            top_score = total_score
            top_seed = seed
        n -= 1
    min_score, score_avg, total_score, seed, cluster_centroids, final_labels, outputs = allocator(top_seed)
    print(f"Seed: {top_seed} | Min Score: {min_score:.4f} | Avg Score: {score_avg:.4f} | Total Score: {total_score:.4f}")

    # ── Plotting ───────────────────────────────────────────────
    plt.figure(figsize=(10, 8))

    # Convert cluster names to numeric indices for coloring
    cluster_to_idx = {c: i for i, c in enumerate(clusters)}
    point_colors = [cluster_to_idx[c] for c in outputs]

    # Scatter points
    scatter = plt.scatter(X[:, 0], X[:, 1], 
                        c=point_colors, cmap='tab10', 
                        s=80)

    # Add point index labels
    for i in range(len(X)):
        plt.text(X[i, 0], X[i, 1], str(final_labels[i]), fontsize=9, ha='right')

    # Plot centroids
    centroids_array = np.array(cluster_centroids)
    plt.scatter(centroids_array[:, 0], centroids_array[:, 1], 
                marker='x', s=100, c='black', 
                label='Centroids')

    # ── Plot PCA Principal Direction Line ──────────────────────

    # Make a long line along PCA direction
    line_length = 0.01 
    pca_line_x = [
        pca.mean_[0] - line_length * principal_direction[0],
        pca.mean_[0] + line_length * principal_direction[0]
    ]
    pca_line_y = [
        pca.mean_[1] - line_length * principal_direction[1],
        pca.mean_[1] + line_length * principal_direction[1]
    ]

    plt.plot(pca_line_x, pca_line_y, 
            color='red', linewidth=2, 
            label='PCA Direction')

    # ── Legend ────────────────────────────────────────────────
    handles = []
    for c in clusters:
        handles.append(
            mpatches.Patch(color=plt.cm.tab10(cluster_to_idx[c]),
                        label=c)
        )

    plt.legend(handles=handles + [mpatches.Patch(color='black', label='Centroids')],
            loc='best')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Clustered Points with Centroids and PCA Direction")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()