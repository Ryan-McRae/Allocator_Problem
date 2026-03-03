# Cell Tower Frequency Allocation

### Technical Assessment — BI & Development

---

## 1. Problem Definition

The objective is to assign frequencies (labels) to cell towers in 2D space such that towers sharing the same frequency are as far apart as possible, minimising radio interference. At its core, this is a **spatial labelling problem**: given a set of points, assign labels such that similarly-labelled points are maximally separated.

---

## 2. Methodology

### 2.1 Building Intuition from 1D

To develop the approach, I first simplified the problem to one dimension with two labels. In this case, the optimal strategy is clear: sort the points along the line and alternate labels. This guarantees maximum separation between same-label points.

The key insight this gives us is:

- **Cluster the closest points together**, then **label within clusters** such that no two points in the same cluster share a label.
- By labelling consistently in the same direction across all clusters, adjacent cluster boundaries also receive different labels.

### 2.2 Extrapolating to 2D — The Labelling Direction

Extending this to 2D raises a question: what direction should we use to order points for labelling? The direction cannot be arbitrary — it needs to reflect the axis along which the data is most spread out, in order to maximise separation between same-label towers.

Rather than testing every possible direction, I used **Principal Component Analysis (PCA)**, which finds the direction of greatest variance in the data. Projecting all points onto the first principal component gives a meaningful 1D ordering that captures the dominant spatial structure of the tower layout.

### 2.3 Clustering with K-Means

With 19 towers and 6 available frequencies (110–115), I set the number of clusters to:

$$k = \lfloor N / \text{labels} \rfloor = \lfloor 19 / 6 \rfloor \approx 4$$

This ensures each cluster contains roughly 4–5 towers. Since there are 6 available frequencies and at most ~5 towers per cluster, every tower in a cluster can be assigned a **unique frequency**, guaranteeing no two close neighbors share a label.

K-Means was implemented with both random and pre-defined initial seeds to mitigate sensitivity to initialisation. The best-performing seed was selected based on the scoring metric described below.

### 2.4 Frequency Assignment

Within each cluster, towers are projected onto the global first principal component. Frequencies are then assigned in **ascending order of projection value** using a round-robin scheme. This ensures:

- No two towers in the same cluster share a frequency.
- Across cluster boundaries, adjacent towers receive different frequencies due to the consistent global ordering direction.

---

## 3. Scoring Metric

To evaluate and compare solutions, a two-part scoring metric was defined over all pairs of towers sharing the same frequency:

| Metric           | Description                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------ |
| **Min Distance** | Smallest distance between any two same-frequency towers — the worst-case interference risk |
| **Avg Distance** | Mean distance between all same-frequency tower pairs — overall spatial spread              |
| **Hard Score**   | Min Distance × Avg Distance — the final combined score                                     |

Multiplying the two scores penalises solutions that perform well on average but have a single dangerously close same-frequency pair. A high hard score requires both good worst-case and good average separation.

---

## 4. Results

The allocator was benchmarked against a random assignment baseline (run 100,000 times) and against a graph coloring solution for comparison.

| Method                           | Min Distance (m) | Avg Distance (m) | Hard Score     |
| -------------------------------- | ---------------- | ---------------- | -------------- |
| Random Assignment                | ~114             | ~397             | ~45,258        |
| **KMeans + PCA (this approach)** | **337.58**       | **675.71**       | **228,107.68** |
| Graph Coloring                   | 344.75           | 729.40           | 252,541.24     |

The KMeans + PCA approach outperforms random assignment by approximately **5×**. It achieves **90% of the graph coloring score** despite being a significantly simpler algorithm — graph coloring requires explicit interference graph construction and a threshold sweep across 80 candidates, whereas this approach requires only clustering and a single PCA decomposition.

---

## 5. Discussion

The primary strength of this approach is its **conceptual simplicity and interpretability**. The intuition maps directly from the 1D case, and the use of PCA as a labelling axis is a principled choice grounded in the geometry of the data.

The main limitation is that K-Means cluster boundaries are soft — two towers that are close to each other can fall on opposite sides of a boundary and accidentally receive the same frequency. Graph coloring explicitly prevents this by modelling every pairwise proximity constraint as a hard edge. This is likely the source of the ~10% gap in scores.

Future improvements could include enforcing a maximum cluster size to prevent unbalanced splits, or adding a post-processing step that detects and resolves any same-frequency boundary conflicts.

---

## 6. Code Overview

| Section                | Description                                                                    |
| ---------------------- | ------------------------------------------------------------------------------ |
| **Configuration**      | Data loading and parameter setup (number of clusters, labels, random state)    |
| **PCA**                | First principal component extraction via scikit-learn                          |
| **Allocator Function** | K-Means with configurable initial seeds; convergence via centroid reassignment |
| **Cluster Grouping**   | Nested index lists of towers per cluster                                       |
| **Projection**         | Per-cluster projection of towers onto the first principal component            |
| **Labelling**          | Round-robin frequency assignment ordered by projection value                   |
| **Scoring**            | Hard score computation for seed selection and benchmarking                     |
| **Main**               | Runs allocator over _n_ seeds, selects best, plots results                     |

---

_Source code and performance history available on GitHub._
