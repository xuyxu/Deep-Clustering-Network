import numpy as np
from joblib import Parallel, delayed


def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
    return dis_mat


class KMeans(object):
    
    def __init__(self, args):
        self.args = args
        self.n_features = args.latent_dim
        self.n_clusters = args.n_clusters
        self.clusters = np.zeros((self.n_clusters, self.n_features))
        self.n_jobs = args.n_jobs
    
    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_distance)(X, self.clusters[i])
            for i in range(self.n_clusters))
        dis_mat = np.hstack(dis_mat)
        
        return dis_mat
    
    def init_cluster(self, X, indices=None):
        n_samples = X.shape[0]
        if indices is None:
            indices = np.random.choice(n_samples, self.n_clusters, 
                                       replace=False)
        self.clusters += X[indices, :]
    
    def update_cluster(self, X, cluster_idx):
        n_samples = X.shape[0]
        for i in range(n_samples):
            diff = self.clusters[cluster_idx] - X[i]
            self.clusters[cluster_idx] -= diff / n_samples
        
    def update_assign(self, X):
        dis_mat = self._compute_dist(X)
        
        return np.argmin(dis_mat, axis=1)