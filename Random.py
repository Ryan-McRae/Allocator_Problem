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
# ── Assignment function ──────────────────────────────
def randomiser():
    # ── Points Based on Position  ───────────────────────────────────────────────
    final_labels = [None] * len(x1)
    for i in range(len(X)):
        final_labels[i] = labels[random.randint(0, len(labels)-1)]
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
    
    return min_score, score_avg, total_score, final_labels

n = 100000
top_score = 0
top_seed = None
while n > 0:
    min_score, score_avg, total_score, final_labels = randomiser()
    #print(f"Seed: {seed} | Min Score: {min_score:.4f} | Avg Score: {score_avg:.4f} | Total Score: {total_score:.4f}")
    if total_score > top_score:
        top_score = total_score
        top_seed = final_labels
    n -= 1
#min_score, score_avg, total_score, final_labels = randomiser(top_seed)
print(f"Seed: {top_seed} | Min Score: {min_score:.4f} | Avg Score: {score_avg:.4f} | Total Score: {total_score:.4f}")
final_labels = top_seed 
# ── Plotting ───────────────────────────────────────────────
plt.figure(figsize=(10, 8))

# Map each label ("110", "111", etc.) to a numeric color
label_to_idx = {label: i for i, label in enumerate(labels)}

# Use the assigned label for each point
point_colors = [label_to_idx[c] for c in final_labels]

# Scatter points
plt.scatter(X[:, 0], X[:, 1],
            c=point_colors,
            cmap='tab10',
            s=100)

# Add label text next to each point
for i in range(len(X)):
    plt.text(X[i, 0], X[i, 1],
             final_labels[i],
             fontsize=9,
             ha='right')

# Legend
handles = [
    mpatches.Patch(color=plt.cm.tab10(label_to_idx[l]),
                   label=l)
    for l in labels
]

plt.legend(handles=handles, loc='best')

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Random Label Assignment")
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()