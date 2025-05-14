import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist, squareform

class HierarchicalClusteringClusterization:
    def __init__(self, linkage='single', distance_metric='euclidean'):
        self.linkage = linkage  # 'single', 'complete', 'average'
        self.distance_metric = distance_metric
        self.linkage_matrix = None
        
    def _calculate_distance_matrix(self, X):
        """Обчислення матриці відстаней"""
        n = len(X)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                if self.distance_metric == 'euclidean':
                    dist = np.sqrt(np.sum((X[i] - X[j])**2))
                elif self.distance_metric == 'manhattan':
                    dist = np.sum(np.abs(X[i] - X[j]))
                
                distances[i, j] = distances[j, i] = dist
                
        return distances
    
    def _find_closest_clusters(self, distances):
        """Знаходження найближчих кластерів"""
        min_dist = np.inf
        merge_i, merge_j = -1, -1
        
        n = len(distances)
        for i in range(n):
            for j in range(i+1, n):
                if distances[i, j] < min_dist:
                    min_dist = distances[i, j]
                    merge_i, merge_j = i, j
                    
        return merge_i, merge_j, min_dist

    def _update_distances(self, distances, clusters, merge_i, merge_j):
        n = len(distances)
        new_distances = np.full((n - 1, n - 1), np.inf)

        indices_map = [idx for idx in range(n) if idx != merge_j]

        for i_new, i_old in enumerate(indices_map):
            for j_new, j_old in enumerate(indices_map):
                if j_new <= i_new:
                    continue

                if i_old == merge_i:
                    if self.linkage == 'single':
                        dist = min(distances[merge_i, j_old], distances[merge_j, j_old])
                    elif self.linkage == 'complete':
                        dist = max(distances[merge_i, j_old], distances[merge_j, j_old])
                    elif self.linkage == 'average':
                        dist = (distances[merge_i, j_old] + distances[merge_j, j_old]) / 2
                elif j_old == merge_i:
                    if self.linkage == 'single':
                        dist = min(distances[merge_i, i_old], distances[merge_j, i_old])
                    elif self.linkage == 'complete':
                        dist = max(distances[merge_i, i_old], distances[merge_j, i_old])
                    elif self.linkage == 'average':
                        dist = (distances[merge_i, i_old] + distances[merge_j, i_old]) / 2
                else:
                    dist = distances[i_old, j_old]

                new_distances[i_new, j_new] = new_distances[j_new, i_new] = dist

        return new_distances

    def fit(self, X):
        """Виконання ієрархічної кластеризації"""
        n = len(X)
        distances = self._calculate_distance_matrix(X)
        
        clusters = [[i] for i in range(n)]
        cluster_ids = list(range(n))
        self.linkage_matrix = []

        current_cluster_id = n

        while len(clusters) > 1:
            merge_i, merge_j, min_dist = self._find_closest_clusters(distances)

            new_cluster = clusters[merge_i] + clusters[merge_j]
            cluster_size = len(new_cluster)

            self.linkage_matrix.append([
                cluster_ids[merge_i],
                cluster_ids[merge_j],
                min_dist,
                cluster_size
            ])

            if merge_i > merge_j:
                merge_i, merge_j = merge_j, merge_i

            clusters[merge_i] = new_cluster
            cluster_ids[merge_i] = current_cluster_id
            clusters.pop(merge_j)
            cluster_ids.pop(merge_j)

            current_cluster_id += 1

            distances = self._update_distances(distances, clusters, merge_i, merge_j)

        self.linkage_matrix = np.array(self.linkage_matrix)
        return self

    def get_clusters(self, n_clusters, X):
        if self.linkage_matrix is None:
            raise ValueError("Потрібно спочатку виконати fit()")

        if n_clusters >= len(X):
            return list(range(len(X)))

        clusters = {i: [i] for i in range(len(X))}
        next_cluster_id = len(X)

        for i, (left_id, right_id, dist, _) in enumerate(self.linkage_matrix):
            if len(clusters) <= n_clusters:
                break
            
            left_id, right_id = int(left_id), int(right_id)
            new_cluster = clusters[left_id] + clusters[right_id]

            del clusters[left_id]
            del clusters[right_id]

            clusters[next_cluster_id] = new_cluster
            next_cluster_id += 1

        labels = np.zeros(len(X), dtype=int)
        for cluster_idx, cluster_points in enumerate(clusters.values()):
            for point_idx in cluster_points:
                labels[point_idx] = cluster_idx

        return labels



    def visualize_dendrogram(self, title="Hierarchical Clustering Dendrogram"):
        """Візуалізація дендрограми"""
        if self.linkage_matrix is None:
            raise ValueError("Потрібно спочатку виконати fit()")
        
        plt.figure(figsize=(12, 8))
        dendrogram(self.linkage_matrix)
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()
    
    def visualize_clusters(self, X, n_clusters, title="Hierarchical Clustering"):
        """Візуалізація кластерів"""
        labels = self.get_clusters(n_clusters, X)
        
        plt.figure(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[i % len(colors)], alpha=0.6, label=f'Cluster {i+1}')
        
        plt.title(title)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_linkage_matrix(self):
        return self.linkage_matrix