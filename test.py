import numpy as np
from scipy.ndimage import sobel
from sklearn.cluster import KMeans
import pyvista as pv

def compute_gradient_tensor(u, v, w, dx=1.0):
    gradients = {}
    for comp, field in zip(['u', 'v', 'w'], [u, v, w]):
        gradients[f"{comp}_x"] = sobel(field, axis=0) / dx
        gradients[f"{comp}_y"] = sobel(field, axis=1) / dx
        gradients[f"{comp}_z"] = sobel(field, axis=2) / dx
    return gradients

def compute_strain_and_rotation(H):
    E = {}
    W = {}
    for i, comp1 in enumerate(['x', 'y', 'z']):
        for j, comp2 in enumerate(['x', 'y', 'z']):
            Hij = H[f"{['u', 'v', 'w'][i]}_{comp2}"]
            Hji = H[f"{['u', 'v', 'w'][j]}_{comp1}"]
            E[f"{comp1}{comp2}"] = 0.5 * (Hij + Hji)
            W[f"{comp1}{comp2}"] = 0.5 * (Hij - Hji)
    return E, W

def extract_features(u, v, w, dx=1.0):
    H = compute_gradient_tensor(u, v, w, dx)
    E, W = compute_strain_and_rotation(H)

    # Compute feature vectors
    features = []
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            for k in range(u.shape[2]):
                strain_magnitude = np.sqrt(sum(E[key][i, j, k]**2 for key in E.keys()))
                rotation_magnitude = np.sqrt(sum(W[key][i, j, k]**2 for key in W.keys()))
                gradient_magnitude = np.sqrt(sum(sobel(E[key], axis=axis)[i, j, k]**2 for key in E.keys() for axis in range(3)))
                displacement_magnitude = np.sqrt(u[i, j, k]**2 + v[i, j, k]**2 + w[i, j, k]**2)
                features.append([strain_magnitude, rotation_magnitude, gradient_magnitude, displacement_magnitude])
    return np.array(features)

def cluster_displacement_vectors(features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels, kmeans

def visualize_clusters(u, v, w, labels, grid_shape):
    grid = pv.UniformGrid()
    grid.dimensions = np.array(grid_shape) + 1
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)
    grid["Cluster"] = labels.reshape(grid_shape).astype(np.float32)
    grid.plot(scalars="Cluster", cmap="viridis", show_edges=True, title="Clustered Displacement Vectors")
    pass

# Example 3D displacement field
x, y, z = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30), np.linspace(0, 1, 30))
u = 0.1 * x + 0.02 * y**2
v = 0.05 * y + 0.01 * z**2
w = 0.03 * z + 0.01 * x**2

# Extract features
features = extract_features(u, v, w)

# Cluster features
labels, kmeans = cluster_displacement_vectors(features, n_clusters=3)

# Visualize clustering
visualize_clusters(u, v, w, labels, grid_shape=u.shape)