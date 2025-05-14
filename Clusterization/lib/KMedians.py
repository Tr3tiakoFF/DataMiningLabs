import numpy as np
import matplotlib.pyplot as plt

class KMediansClusterization:
    def __init__(self, k, max_iters=100, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        
    def fit(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
        self.medians = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iters):
            distances = np.sum(np.abs(X - self.medians[:, np.newaxis]), axis=2)
            self.labels = np.argmin(distances, axis=0)
            
            new_medians = np.array([np.median(X[self.labels == i], axis=0) for i in range(self.k)])
            
            if np.allclose(self.medians, new_medians):
                break
                
            self.medians = new_medians
            
        return self
    
    def predict(self, X):
        distances = np.sum(np.abs(X - self.medians[:, np.newaxis]), axis=2)
        return np.argmin(distances, axis=0)
    
    def fit_predict(self, X):
        return self.fit(X).labels
    
    def visualize(self, X, title="K-Medians Clustering"):
        plt.figure(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[i % len(colors)], alpha=0.6, label=f'Cluster {i+1}')
        
        plt.scatter(self.medians[:, 0], self.medians[:, 1], 
                   c='black', marker='D', s=200, linewidths=3, label='Medians')
        
        plt.title(title)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_medians(self):
        return self.medians
    
    def get_labels(self):
        return self.labels