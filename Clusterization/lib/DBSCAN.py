import numpy as np
import matplotlib.pyplot as plt

class DBSCANClusterization:
    def __init__(self, eps=0.5, min_samples=5, distance_metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.distance_metric = distance_metric
        self.labels = None
        
    def _calculate_distance(self, x1, x2):
        """Обчислення відстані між двома точками"""
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2)**2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Підтримувані метрики: 'euclidean', 'manhattan'")
    
    def _get_neighbors(self, X, point_idx):
        """Знаходження сусідів точки в межах eps"""
        neighbors = []
        point = X[point_idx]
        
        for i, neighbor in enumerate(X):
            if self._calculate_distance(point, neighbor) <= self.eps:
                neighbors.append(i)
        
        return neighbors
    
    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited, labels):
        """Розширення кластера"""
        labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = self._get_neighbors(X, neighbor_idx)
                
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors.extend([n for n in neighbor_neighbors if n not in neighbors])
            
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
                
            i += 1
    
    def fit(self, X):
        """Виконання DBSCAN кластеризації"""
        X = np.array(X)
        n_points = len(X)
        
        visited = [False] * n_points
        labels = [-1] * n_points
        cluster_id = 0
        
        for point_idx in range(n_points):
            if visited[point_idx]:
                continue
                
            visited[point_idx] = True
            neighbors = self._get_neighbors(X, point_idx)
            
            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1
            else:
                self._expand_cluster(X, point_idx, neighbors, cluster_id, visited, labels)
                cluster_id += 1
        
        self.labels = np.array(labels)
        return self
    
    def fit_predict(self, X):
        """Виконання кластеризації та повернення міток"""
        return self.fit(X).labels
    
    def get_cluster_info(self):
        """Отримання інформації про кластери"""
        if self.labels is None:
            raise ValueError("Спочатку виконайте fit()")
        
        unique_labels = np.unique(self.labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = np.sum(self.labels == -1)
        
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[f'Cluster {label+1}'] = np.sum(self.labels == label)
        
        return {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'cluster_sizes': cluster_sizes
        }
    
    def get_labels(self):
        return self.labels
    
    def predict(self, X_new, X_train):
        """Прогнозування для нових точок на основі навчальних даних"""
        if self.labels is None:
            raise ValueError("Спочатку виконайте fit()")
        
        X_new = np.array(X_new)
        X_train = np.array(X_train)
        predictions = []
        
        for new_point in X_new:
            distances = [self._calculate_distance(new_point, train_point) 
                        for train_point in X_train]
            
            neighbors_in_eps = []
            for i, dist in enumerate(distances):
                if dist <= self.eps:
                    neighbors_in_eps.append(i)
            
            if len(neighbors_in_eps) == 0:
                predictions.append(-1)
            else:
                neighbor_labels = [self.labels[i] for i in neighbors_in_eps]
                non_noise_labels = [label for label in neighbor_labels if label != -1]
                
                if len(non_noise_labels) == 0:
                    predictions.append(-1)
                else:
                    from collections import Counter
                    most_common = Counter(non_noise_labels).most_common(1)[0][0]
                    predictions.append(most_common)
        
        return np.array(predictions)
    
    def get_core_samples(self, X):
        """Повертає індекси основних точок (core samples)"""
        if self.labels is None:
            raise ValueError("Спочатку виконайте fit()")
        
        core_indices = []
        for idx in range(len(X)):
            neighbors = self._get_neighbors(X, idx)
            if len(neighbors) >= self.min_samples:
                core_indices.append(idx)
        
        return core_indices
    
    def visualize(self, X, title="DBSCAN Clustering"):
        """Візуалізація результатів кластеризації"""
        if self.labels is None:
            raise ValueError("Спочатку виконайте fit()")
        
        X = np.array(X)
        plt.figure(figsize=(12, 8))
        
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan']
        
        unique_labels = np.unique(self.labels)
        
        for label in unique_labels:
            if label == -1:
                noise_points = X[self.labels == label]
                plt.scatter(noise_points[:, 0], noise_points[:, 1], 
                           c='black', marker='x', s=50, alpha=0.8, label='Noise')
            else:
                cluster_points = X[self.labels == label]
                color = colors[label % len(colors)]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                           c=color, alpha=0.7, s=60, label=f'Cluster {label+1}')
        
        core_samples = self.get_core_samples(X)
        if core_samples:
            core_points = X[core_samples]
            plt.scatter(core_points[:, 0], core_points[:, 1], 
                       facecolors='none', edgecolors='black', s=100, 
                       linewidth=2, alpha=0.8, label='Core Points')
        
        plt.title(f"{title}\neps={self.eps}, min_samples={self.min_samples}")
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()