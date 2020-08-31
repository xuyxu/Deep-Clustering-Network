import pytest
import argparse
import numpy as np
from clustering import Clustering


def test_core():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-centroids", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    
    data = np.array([
                        [0.0, 0.0],
                        [2.0, 2.0],
                        [4.0, 4.0],
                        [6.0, 6.0]
                    ])
    
    model = Clustering(args)
    model.init_centroid(data, indices=np.array([0, 3]))
    
    # Update the assignment
    assign_res = model.update_assign(data)
    assert np.array_equal(assign_res, np.array([0, 0, 1, 1]))
    
    # Update the centroids
    model.update_centroid(data, assign_res)
    assert np.array_equal(model.centroids,
                          np.array([[1.0, 1.0],
                                    [5.0, 5.0]]))


if __name__ == '__main__':
    
    pytest.main([__file__])
