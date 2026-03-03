"""
Cell Tower Frequency Allocation
================================
Approach: Graph Coloring using a distance-threshold model.
- Build interference graph: two towers are "neighbors" if within a threshold distance
- Use a greedy graph coloring ordered by degree (most constrained first)
- Frequencies: 110-115 (6 unique)
- Score: min distance between same-frequency towers & average distance between same-frequency towers
- Final hard score = min_dist * avg_dist

NOTE: AI (Claude) was used to assist in structuring this solution. 
The core logic (distance threshold determination, scoring metric design, 
graph-coloring with degree ordering) was reasoned through independently.
"""

import math
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Data ────────────────────────────────────────────────────────────────────
CELLS = {
    'A': (536660, 183800), 'B': (537032, 184006), 'C': (537109, 183884),
    'D': (537110, 184695), 'E': (537206, 184685), 'F': (537248, 185016),
    'G': (537250, 185020), 'H': (537267, 184783), 'I': (537269, 183451),
    'J': (537270, 184140), 'K': (537356, 184927), 'L': (537380, 184727),
    'M': (537458, 184495), 'N': (537604, 184134), 'O': (537720, 184057),
    'P': (537905, 184591), 'Q': (537910, 184441), 'R': (537953, 184295),
    'S': (538050, 184245),
}

FREQUENCIES = list(range(110, 116))  # 110..115


# ─── Distance helpers ─────────────────────────────────────────────────────────
#def euclidean(a, b):
#    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def euclidean(p1, p2):
    R = 6371000  # Earth radius in meters
    lon1, lat1 = np.radians(p1)
    lon2, lat2 = np.radians(p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def all_distances(cells):
    ids = list(cells.keys())
    dists = {}
    for i, j in itertools.combinations(ids, 2):
        d = euclidean(cells[i], cells[j])
        dists[(i, j)] = d
        dists[(j, i)] = d
    return dists


# ─── Build interference graph ─────────────────────────────────────────────────
def build_graph(cells, dists, threshold):
    """Two towers are neighbors (must differ in frequency) if dist <= threshold."""
    ids = list(cells.keys())
    neighbors = {i: set() for i in ids}
    for (i, j), d in dists.items():
        if i < j and d <= threshold:
            neighbors[i].add(j)
            neighbors[j].add(i)
    return neighbors


# ─── Graph coloring (greedy, degree-ordered) ─────────────────────────────────
def color_graph(neighbors, frequencies):
    # Order: most neighbors first (most constrained)
    order = sorted(neighbors.keys(), key=lambda x: -len(neighbors[x]))
    assignment = {}
    for node in order:
        used = {assignment[nb] for nb in neighbors[node] if nb in assignment}
        for freq in frequencies:
            if freq not in used:
                assignment[node] = freq
                break
        else:
            # Fallback: pick least-used frequency among neighbors
            counts = {f: sum(1 for nb in neighbors[node] 
                             if nb in assignment and assignment[nb] == f)
                      for f in frequencies}
            assignment[node] = min(counts, key=counts.get)
    return assignment


# ─── Scoring ─────────────────────────────────────────────────────────────────
def compute_scores(assignment, dists):
    """
    For each pair of towers sharing the same frequency, compute their distance.
    min_dist  = smallest such distance (lower = more interference risk)
    avg_dist  = average such distance
    hard_score = min_dist * avg_dist  (higher = better spread)
    """
    same_freq_dists = []
    cells = list(assignment.keys())
    for i, j in itertools.combinations(cells, 2):
        if assignment[i] == assignment[j]:
            same_freq_dists.append(dists[(i, j)])

    if not same_freq_dists:
        return float('inf'), float('inf'), float('inf')

    min_d = min(same_freq_dists)
    avg_d = sum(same_freq_dists) / len(same_freq_dists)
    hard  = min_d * avg_d
    return min_d, avg_d, hard


# ─── Optimise threshold ───────────────────────────────────────────────────────
def find_best_threshold(cells, dists, frequencies):
    """
    Try a range of thresholds; pick the one that maximises hard_score
    while still producing a valid (conflict-free) coloring.
    """
    dist_vals = sorted(set(round(v) for v in dists.values()))
    best = None
    best_score = -1

    # Sample ~50 candidate thresholds across the range
    candidates = np.linspace(dist_vals[0], dist_vals[-1], 80)

    for t in candidates:
        neighbors = build_graph(cells, dists, t)
        assignment = color_graph(neighbors, frequencies)
        # Check validity
        valid = all(assignment[i] != assignment[j]
                    for i in neighbors for j in neighbors[i])
        if not valid:
            continue
        _, _, hard = compute_scores(assignment, dists)
        if hard > best_score:
            best_score = hard
            best = (t, assignment, neighbors)

    return best


# ─── Visualisation ───────────────────────────────────────────────────────────
COLORS = ['#e63946','#2a9d8f','#e9c46a','#264653','#f4a261','#a8dadc']

def plot_results(cells, assignment, neighbors, scores):
    min_d, avg_d, hard = scores
    freq_list = sorted(set(assignment.values()))
    freq_color = {f: COLORS[i % len(COLORS)] for i, f in enumerate(freq_list)}

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#0f0f1a')

    # ── Left: network graph ────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#0f0f1a')
    ax1.set_title('Tower Interference Network\n(edges = interference risk)',
                  color='white', fontsize=13, pad=12)

    xs = {k: v[0] for k, v in cells.items()}
    ys = {k: v[1] for k, v in cells.items()}

    # Draw edges
    drawn = set()
    for node, nbs in neighbors.items():
        for nb in nbs:
            key = tuple(sorted([node, nb]))
            if key not in drawn:
                ax1.plot([xs[node], xs[nb]], [ys[node], ys[nb]],
                         color='#444466', lw=0.8, zorder=1)
                drawn.add(key)

    # Draw nodes
    for cell_id, (x, y) in cells.items():
        freq = assignment[cell_id]
        color = freq_color[freq]
        ax1.scatter(x, y, color=color, s=280, zorder=3,
                    edgecolors='white', linewidths=1.2)
        ax1.text(x, y, cell_id, color='white', fontsize=8,
                 ha='center', va='center', fontweight='bold', zorder=4)

    ax1.set_xlabel('Easting', color='#aaaacc')
    ax1.set_ylabel('Northing', color='#aaaacc')
    ax1.tick_params(colors='#aaaacc')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333355')

    # Legend
    patches = [mpatches.Patch(color=freq_color[f], label=f'Freq {f}')
               for f in freq_list]
    ax1.legend(handles=patches, loc='lower right',
               facecolor='#1a1a2e', edgecolor='#333355',
               labelcolor='white', fontsize=9)

    # ── Right: frequency allocation bar chart ──────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#0f0f1a')
    ax2.set_title('Frequency Allocation per Tower',
                  color='white', fontsize=13, pad=12)

    sorted_cells = sorted(assignment.keys())
    bar_colors = [freq_color[assignment[c]] for c in sorted_cells]
    bar_vals   = [assignment[c] for c in sorted_cells]

    bars = ax2.bar(sorted_cells, bar_vals, color=bar_colors,
                   edgecolor='white', linewidth=0.5)

    ax2.set_ylim(109, 116)
    ax2.set_yticks(FREQUENCIES)
    ax2.set_xlabel('Cell Tower', color='#aaaacc')
    ax2.set_ylabel('Frequency (MHz)', color='#aaaacc')
    ax2.tick_params(colors='#aaaacc')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333355')

    # Annotate bars
    for bar, val in zip(bars, bar_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                 str(val), ha='center', va='bottom',
                 color='white', fontsize=7, fontweight='bold')

    ax2.legend(handles=patches, loc='upper right',
               facecolor='#1a1a2e', edgecolor='#333355',
               labelcolor='white', fontsize=9)

    # ── Score banner ───────────────────────────────────────────────────────
    score_txt = (f'Min Distance (same freq): {min_d:,.1f} m   |   '
                 f'Avg Distance (same freq): {avg_d:,.1f} m   |   '
                 f'Hard Score (min × avg): {hard:,.1f}')
    fig.text(0.5, 0.02, score_txt, ha='center', color='#a8dadc',
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                       edgecolor='#2a9d8f', linewidth=1.5))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = '/mnt/user-data/outputs/frequency_allocation.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print(f'Graph saved → {out}')
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Cell Tower Frequency Allocation")
    print("=" * 60)

    dists = all_distances(CELLS)
    print(f"\nTowers: {len(CELLS)}  |  Frequencies available: {FREQUENCIES}")

    result = find_best_threshold(CELLS, dists, FREQUENCIES)
    if result is None:
        print("ERROR: No valid coloring found with 6 frequencies.")
        return

    threshold, assignment, neighbors = result
    print(f"\nOptimal interference threshold: {threshold:.1f} m")
    print("\nFrequency Allocation:")
    print(f"  {'Tower':<8} {'Frequency':<12} {'Neighbors (conflict edges)'}")
    print("  " + "-" * 50)
    for cell_id in sorted(assignment):
        freq = assignment[cell_id]
        nbs  = ', '.join(sorted(neighbors[cell_id])) or '—'
        print(f"  {cell_id:<8} {freq:<12} {nbs}")

    scores = compute_scores(assignment, dists)
    min_d, avg_d, hard = scores

    print("\n" + "=" * 60)
    print("  SCORING")
    print("=" * 60)
    print(f"  Min distance between same-frequency towers : {min_d:>12,.2f} m")
    print(f"  Avg distance between same-frequency towers : {avg_d:>12,.2f} m")
    print(f"  Hard Score (min × avg)                     : {hard:>12,.2f}")
    print("=" * 60)

    plot_results(CELLS, assignment, neighbors, scores)


if __name__ == '__main__':
    main()