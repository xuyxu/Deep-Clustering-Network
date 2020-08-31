import numpy as np
from joblib import Parallel, delayed


def _parallel_compute_distance(X, centroid):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - centroid) ** 2, axis=0))
    return dis_mat


class Clustering(object):
    
    def __init__(self, args):
        super(Clustering, self).__init__()
        self.args = args
        self.n_centroids = args.n_centroids
        self.latent_dim = args.latent_dim
        self.centroids = np.zeros((self.n_centroids, self.latent_dim))
        self.n_jobs = args.n_jobs
    
    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_distance)(X, self.centroids[i])
            for i in range(self.n_centroids))
        dis_mat = np.hstack(dis_mat)
        
        return dis_mat
    
    def init_centroid(self, latent_X, indices=None):
        n_samples = latent_X.shape[0]
        if indices is None:
            indices = np.random.choice(n_samples, self.n_centroids, 
                                       replace=False)
        self.centroids += latent_X[indices, :]
    
    def update_centroid(self, latent_X, centroid_id):
        for i in range(self.n_centroids):
            mask = centroid_id == i

            # Avoid empty slicing
            if np.all(mask == False):
                continue
            
            self.centroids[i] = np.mean(latent_X[mask, :], axis=0)
        
    def update_assign(self, latent_X):
        dis_mat = self._compute_dist(latent_X)
        
        return np.argmin(dis_mat, axis=1)
        
    
if __name__ == '__main__':
    
    import time
    import argparse
    from sklearn.datasets import load_iris
    
    data = load_iris().data
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-centroids", type=int, default=3)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    
    model = Clustering(args)
    model.init_centroid(data)
    
    for _ in range(100):
        
        tic = time.time()
        centroid_id = model.update_assign(data)
        model.update_centroid(data, centroid_id)
        toc = time.time()
        print('Time per iteration: {:.5f}s'.format(toc - tic))
