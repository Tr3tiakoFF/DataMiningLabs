import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class NearestNeighborClusterization:
    def __init__(self, n_neighbors=3, distance_metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def _calculate_distance(self, x1, x2):
        """Обчислення відстані між двома точками"""
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2)**2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'cosine':
            dot_product = np.dot(x1, x2)
            norm_x1 = np.linalg.norm(x1)
            norm_x2 = np.linalg.norm(x2)
            return 1 - (dot_product / (norm_x1 * norm_x2))
        else:
            raise ValueError("Підтримувані метрики: 'euclidean', 'manhattan', 'cosine'")
    
    def fit(self, X, y=None):
        """Навчання моделі (збереження тренувальних даних)"""
        self.X_train = np.array(X)
        self.y_train = np.array(y) if y is not None else None
        return self
    
    def predict(self, X):
        """Прогнозування для нових точок"""
        if self.X_train is None:
            raise ValueError("Модель не навчена. Викличте fit() спочатку.")
        
        X = np.array(X)
        predictions = []
        
        for point in X:
            distances = []
            for train_point in self.X_train:
                dist = self._calculate_distance(point, train_point)
                distances.append(dist)
            
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            
            if self.y_train is not None:
                nearest_labels = self.y_train[nearest_indices]
                prediction = Counter(nearest_labels).most_common(1)[0][0]
            else:
                prediction = nearest_indices
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def fit_predict_clustering(self, X, threshold=None):
        """Кластеризація на основі найближчих сусідів"""
        X = np.array(X)
        n_points = len(X)
        
        distance_matrix = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = self._calculate_distance(X[i], X[j])
                distance_matrix[i, j] = distance_matrix[j, i] = dist
        
        if threshold is None:
            all_distances = distance_matrix[distance_matrix > 0]
            threshold = np.percentile(all_distances, 30)
        
        labels = np.arange(n_points)
        
        for i in range(n_points):
            neighbors_distances = [(j, distance_matrix[i, j]) for j in range(n_points) if i != j]
            neighbors_distances.sort(key=lambda x: x[1])
            
            for neighbor_idx, dist in neighbors_distances[:self.n_neighbors]:
                if dist <= threshold:
                    old_label = labels[neighbor_idx]
                    new_label = labels[i]
                    if old_label != new_label:
                        labels[labels == old_label] = new_label
        
        unique_labels = np.unique(labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        final_labels = np.array([label_mapping[label] for label in labels])
        
        self.labels = final_labels
        return final_labels
    
    def get_neighbors(self, point, X=None):
        """Отримання найближчих сусідів для точки"""
        if X is None and self.X_train is None:
            raise ValueError("Потрібно надати дані або навчити модель")
        
        search_data = X if X is not None else self.X_train
        
        distances = []
        for i, train_point in enumerate(search_data):
            dist = self._calculate_distance(point, train_point)
            distances.append((i, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:self.n_neighbors]
    
    def visualize_neighbors(self, query_point, X, title="Nearest Neighbors"):
        """Візуалізація найближчих сусідів"""
        neighbors = self.get_neighbors(query_point, X)
        neighbor_indices = [n[0] for n in neighbors]
        
        plt.figure(figsize=(10, 8))
        
        plt.scatter(X[:, 0], X[:, 1], c='lightblue', alpha=0.6, label='All points')
        
        neighbor_points = X[neighbor_indices]
        plt.scatter(neighbor_points[:, 0], neighbor_points[:, 1], 
                   c='red', s=100, alpha=0.8, label=f'{self.n_neighbors} Nearest Neighbors')
        
        plt.scatter(query_point[0], query_point[1], 
                   c='green', s=200, marker='*', label='Query Point')
        
        for idx in neighbor_indices:
            plt.plot([query_point[0], X[idx, 0]], 
                    [query_point[1], X[idx, 1]], 
                    'k--', alpha=0.5)
        
        plt.title(title)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def visualize_clustering(self, X, title="Nearest Neighbor Clustering"):
        """Візуалізація результатів кластеризації"""
        if not hasattr(self, 'labels'):
            raise ValueError("Спочатку виконайте fit_predict_clustering()")
        
        plt.figure(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        n_clusters = len(np.unique(self.labels))
        for i in range(n_clusters):
            cluster_points = X[self.labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[i % len(colors)], alpha=0.6, label=f'Cluster {i+1}')
        
        plt.title(title)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_labels(self):
        return getattr(self, 'labels', None)