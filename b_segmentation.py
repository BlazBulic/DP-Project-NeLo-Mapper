import os
import sys
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import kmapper as km
from scipy.sparse import load_npz, diags, coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# loads a 3d point cloud from .obj file to np.ndarray(N,3) file
def load_point_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.obj', '.ply', '.off', '.stl']:
        mesh = trimesh.load(path, process=False)
        return np.array(mesh.vertices)
    elif ext == '.npy':
        return np.load(path)
    else:
        raise ValueError(f"Unsupported point cloud extension: {ext}")


# visualizes three segmentation methods side-by-side in 3D.
def visualize_compare(X, labels_nelo, labels_manual, labels_spec, title_nelo = "NeLo Mapper", title_manual = "Manual Mapper", title_spec = "Spectral + KMeans"):
    fig = plt.figure(figsize=(18, 6))

    # NeLo + Mapper
    ax1 = fig.add_subplot(131, projection='3d')
    s1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels_nelo, cmap='tab10', s=5)
    ax1.set_title(title_nelo)
    ax1.axis('off')

    # Manual Laplacian + Mapper
    ax2 = fig.add_subplot(132, projection='3d')
    s2 = ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels_manual, cmap='tab10', s=5)
    ax2.set_title(title_manual)
    ax2.axis('off')

    # Spectral embeddings +  KMeans
    ax3 = fig.add_subplot(133, projection='3d')
    s3 = ax3.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels_spec, cmap='tab10', s=5)
    ax3.set_title(title_spec)
    ax3.axis('off')

    #plt.tight_layout()
    plt.show()


# spectral segmentation with predicted Laplacian matrix
def spectral_segmentation(L_matrix, n_clusters = 2, n_eigen= None):

    if n_eigen is None:
        n_eigen = n_clusters
    
    # compute smallest eigenpairs; the 0th is trivial (constant)
    vals, vecs = eigsh(L_matrix, k=n_eigen+1, which='SM')
    embedding = vecs[:, 1:]  # drop the trivial eigenvector
    
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(embedding)
    return labels, embedding


# Mapper-based segmentation using NeLo Laplacian eigenvectors as filters.
def mapper_segmentation(X, L_matrix, n_filters = 2, min_samp = 10, cover_kwargs=None, clusterer=None):

    # compute smallest eigenpairs; the 0th is trivial (constant)
    vals, vecs = eigsh(L_matrix, k=n_filters+1, which='SM')
    filters = np.vstack([vecs[:, i] for i in range(1, n_filters+1)]).T
    
    # cover & clusterer defaults
    if cover_kwargs is None:
        cover_kwargs = {'n_cubes': 10, 'perc_overlap': 0.3}
    if clusterer is None:
        clusterer = DBSCAN(eps=0.05, min_samples=min_samp)
    
    mapper = km.KeplerMapper(verbose=1)
    cover  = km.Cover(**cover_kwargs)
    
    # construct the Mapper graph
    graph = mapper.map(
        X=X,
        lens=filters,
        cover=cover,
        clusterer=clusterer
    )
    
    # assign labels
    N = X.shape[0]
    labels = -1 * np.ones(N, dtype=int)
    node_list = list(graph['nodes'].keys())
    node_to_idx = {nid: i for i, nid in enumerate(node_list)}
    for nid, pts in graph['nodes'].items():
        li = node_to_idx[nid]
        for pt in pts:
            labels[pt] = li

    return labels, graph


# Mapper-based segmentation using manualy computed Laplacian eigenvectors as filters.
def mapper_segmentation_manual_laplacian(X, sigma = 1.0, n_filters = 2, min_samp=10, cover_kwargs = None, clusterer=None):
    # compute full Gaussian‐kernel adjacency W
    d2 = cdist(X, X, 'sqeuclidean')
    W  = np.exp(-d2 / (2 * sigma**2))
    np.fill_diagonal(W, 0.0)

    # form sparse Laplacian L = D - W
    deg = W.sum(axis=1)
    D   = diags(deg)
    A   = coo_matrix(W)
    L   = D - A

    #  # compute smallest eigenpairs; the 0th is trivial (constant)
    vals, vecs = eigsh(L, k=n_filters+1, which='SM')
    filters = vecs[:, 1:n_filters+1]

    #cover & clusterer defaults
    if cover_kwargs is None:
        cover_kwargs = {'n_cubes': 10, 'perc_overlap': 0.3}
    if clusterer is None:
        clusterer = DBSCAN(eps=0.05, min_samples=min_samp)

    mapper = km.KeplerMapper(verbose=1)
    cover  = km.Cover(**cover_kwargs)

    # construct the Mapper graph
    graph = mapper.map(
        X=X,
        lens=filters,
        cover=cover,
        clusterer=clusterer
    )

    # assign labels
    N = X.shape[0]
    labels = -1 * np.ones(N, dtype=int)
    node_list   = list(graph['nodes'].keys())
    node_to_idx = {nid: i for i, nid in enumerate(node_list)}
    for nid, pts in graph['nodes'].items():
        li = node_to_idx[nid]
        for p in pts:
            labels[p] = li

    return labels, graph


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <obj_path> <laplacian_matrix_path>")
        sys.exit(1)

    obj_path = sys.argv[1]
    laplacian_matrix_path = sys.argv[2]

    # load data
    object_mesh = load_point_cloud(obj_path)
    Laplacian_matrix = load_npz(laplacian_matrix_path)

    # lomparing aproaches
    map_labels_nelo, graph_nelo = mapper_segmentation(object_mesh, Laplacian_matrix, n_filters=4, min_samp=10, cover_kwargs={'n_cubes':10,'perc_overlap':0.3})
    map_labels_manual, graph_manual = mapper_segmentation_manual_laplacian(object_mesh, sigma=0.1, n_filters=4, min_samp=10, cover_kwargs={'n_cubes':10,'perc_overlap':0.3})
    spec_labels, _       = spectral_segmentation(Laplacian_matrix, n_clusters=5)

    print(map_labels_nelo)

    # visualize side by side
    visualize_compare(object_mesh, map_labels_nelo, map_labels_manual, spec_labels)
