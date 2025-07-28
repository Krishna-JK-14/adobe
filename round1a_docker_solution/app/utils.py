from sklearn.cluster import KMeans
import numpy as np

def cluster_font_sizes(font_sizes, n_clusters=3):
    font_sizes = np.array(font_sizes).reshape(-1, 1)
    kmeans = KMeans(n_clusters=min(n_clusters, len(font_sizes)), random_state=0).fit(font_sizes)
    size_to_rank = {}
    for size, label in zip(font_sizes, kmeans.labels_):
        size_to_rank[size[0]] = label
    # Normalize rank (0 = largest)
    sorted_labels = sorted(set(size_to_rank.values()), key=lambda l: -np.mean([s for s, l2 in size_to_rank.items() if l2 == l]))
    label_map = {label: i for i, label in enumerate(sorted_labels)}
    return {size: label_map[label] for size, label in size_to_rank.items()}
