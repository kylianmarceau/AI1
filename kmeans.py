import numpy as np


def initialize_random(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Select k random points from X as initial centroids."""
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()


def initialize_kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Initialize centroids using the K-means++ strategy."""
    n_samples = X.shape[0]
    centroids = np.empty((k, X.shape[1]), dtype=X.dtype)
    centroids[0] = X[rng.integers(n_samples)]
    for i in range(1, k):
        distances = np.min(((X[:, None, :] - centroids[:i]) ** 2).sum(axis=2), axis=1)
        probs = distances / distances.sum()
        cumulative_probs = np.cumsum(probs)
        r = rng.random()
        index = np.searchsorted(cumulative_probs, r)
        centroids[i] = X[index]
    return centroids


def kmeans(
    X: np.ndarray,
    k: int,
    init: str = "random",
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int | None = None,
):
    """Run K-means clustering on data X.

    Parameters
    ----------
    X : np.ndarray
        Data array of shape (n_samples, n_features).
    k : int
        Number of clusters.
    init : str
        Initialization strategy: 'random' or 'kmeans++'.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance for centroid movement.
    random_state : int | None
        Random seed for reproducibility.

    Returns
    -------
    centroids : np.ndarray
        Final centroids of shape (k, n_features).
    labels : np.ndarray
        Cluster assignment for each sample.
    inertia : float
        Final sum of squared distances to cluster centres.
    n_iter : int
        Number of iterations run.
    history : list[float]
        Inertia value at each iteration.
    """

    rng = np.random.default_rng(random_state)
    if init == "random":
        centroids = initialize_random(X, k, rng)
    elif init == "kmeans++":
        centroids = initialize_kmeans_plus_plus(X, k, rng)
    else:
        raise ValueError("init must be 'random' or 'kmeans++'")

    history = []
    for i in range(max_iter):
        # Assign clusters
        distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)

        # Update centroids
        new_centroids = np.empty_like(centroids)
        for j in range(k):
            points = X[labels == j]
            if len(points) == 0:
                new_centroids[j] = X[rng.integers(len(X))]
            else:
                new_centroids[j] = points.mean(axis=0)

        inertia = float(np.sum((X - new_centroids[labels]) ** 2))
        history.append(inertia)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    return centroids, labels, inertia, i + 1, history
