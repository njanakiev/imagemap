import numpy as np
import scipy.linalg
import scipy.spatial


def _calc_dispersion(X, min_r, step_size=0.2, noise=None):
    D = np.zeros(X.shape)
    kdtree = scipy.spatial.cKDTree(X)
    num_neighbors = np.zeros((X.shape[0],))
    
    for curr_idx in range(X.shape[0]):
        idx_list = kdtree.query_ball_point(
            X[curr_idx], min_r)
        num_neighbors[curr_idx] = len(idx_list)
        
        if noise is not None:
            D[curr_idx] += np.random.normal(0, noise, size=(X.shape[1],))
        
        for idx in idx_list:
            dist = scipy.linalg.norm(X[idx] - X[curr_idx])
            if dist > 0.0:
                D[curr_idx] -= step_size * (min_r - dist) * \
                    ((X[curr_idx] - X[idx]) / dist)
    
    return D, num_neighbors


def disperse_points(X, min_r, step_size=0.2, noise=None, iterations=200, verbose=False):
    X_out = X.copy()
    
    for i in range(iterations):
        D, num_neighbors = _calc_dispersion(X_out, min_r, step_size, noise)
        if verbose and i % 20 == 0:
            print(f"Iteration: {i: 5d}, "\
                  f"Max num neighbors: {num_neighbors.max()}")
        
        X_out = X_out - D
    
    return X_out
