import numpy as np
import matplotlib.pyplot as plt

class KMeansClusterization:
    def __init__(self, k, max_iters=100, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        
    def fit(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
            
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iters):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
            
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
            
        return self
    
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def fit_predict(self, X):
        return self.fit(X).labels
    
    def visualize(self, X, title="K-Means Clustering"):
        plt.figure(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[i % len(colors)], alpha=0.6, label=f'Cluster {i+1}')
        
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], 
                   c='black', marker='x', s=200, linewidths=3, label='Centroids')
        
        plt.title(title)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_centroids(self):
        return self.centroids
    
    def get_labels(self):
        return self.labels