import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple


def load_dataset(path: str) -> np.ndarray:
    """Load dataset from CSV; generate synthetic data if not available."""
    csv_path = Path(path)
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # if file is a git-lfs pointer, it will only have few lines/columns
            if df.shape[1] > 1:
                return df.values.astype(float)
        except Exception:
            pass
    # fallback synthetic data
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=-2, scale=0.5, size=(100, 2)),
        rng.normal(loc=0, scale=0.5, size=(100, 2)),
        rng.normal(loc=2, scale=0.5, size=(100, 2)),
    ])
    print("Warning: Using synthetic dataset with three clusters.")
    return X


def initialize_centroids(X: np.ndarray, k: int, rng: np.random.Generator, method: str) -> np.ndarray:
    if method == "random":
        indices = rng.choice(X.shape[0], size=k, replace=False)
        return X[indices]
    elif method == "kmeans++":
        centroids = []
        # choose first centroid randomly
        centroids.append(X[rng.integers(0, X.shape[0])])
        for _ in range(1, k):
            dist_sq = np.min(np.linalg.norm(X[:, None, :] - np.array(centroids)[None, :, :], axis=2) ** 2, axis=1)
            probs = dist_sq / dist_sq.sum()
            centroids.append(X[rng.choice(X.shape[0], p=probs)])
        return np.array(centroids)
    else:
        raise ValueError(f"Unknown init method: {method}")


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def compute_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) == 0:
            centroids[i] = X[np.random.randint(0, X.shape[0])]
        else:
            centroids[i] = cluster_points.mean(axis=0)
    return centroids


def inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    return np.sum((X - centroids[labels]) ** 2)


def kmeans(X: np.ndarray, k: int, init: str, max_iters: int = 300, tol: float = 1e-4, seed: int = None) -> Tuple[np.ndarray, np.ndarray, int, List[float]]:
    rng = np.random.default_rng(seed)
    centroids = initialize_centroids(X, k, rng, init)
    history = []
    for it in range(max_iters):
        labels = assign_clusters(X, centroids)
        current_inertia = inertia(X, labels, centroids)
        history.append(current_inertia)
        new_centroids = compute_centroids(X, labels, k)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break
    labels = assign_clusters(X, centroids)
    history.append(inertia(X, labels, centroids))
    return centroids, labels, it + 1, history


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    # contingency table
    n = len(labels_true)
    labels_true = labels_true.astype(int)
    labels_pred = labels_pred.astype(int)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    contingency = np.zeros((len(classes), len(clusters)), dtype=int)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true == c) & (labels_pred == k))
    sum_comb_c = np.sum([c * (c - 1) // 2 for c in contingency.sum(axis=1)])
    sum_comb_k = np.sum([c * (c - 1) // 2 for c in contingency.sum(axis=0)])
    sum_comb = np.sum([c * (c - 1) // 2 for c in contingency.ravel()])
    total_comb = n * (n - 1) // 2
    expected = sum_comb_c * sum_comb_k / total_comb
    max_index = 0.5 * (sum_comb_c + sum_comb_k)
    return (sum_comb - expected) / (max_index - expected)


def run_experiments(X: np.ndarray, k: int, init: str, runs: int = 5) -> dict:
    results = []
    histories = []
    labels_list = []
    for seed in range(runs):
        centroids, labels, iters, history = kmeans(X, k, init, seed=seed)
        results.append({"seed": seed, "iterations": iters, "final_inertia": history[-1]})
        histories.append(history)
        labels_list.append(labels)
    # compute stability via pairwise ARI
    ari_scores = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            ari_scores.append(adjusted_rand_index(labels_list[i], labels_list[j]))
    stability = float(np.mean(ari_scores)) if ari_scores else float('nan')
    return {"results": results, "histories": histories, "stability": stability}


def plot_histories(histories: List[List[float]], title: str, filename: str):
    plt.figure(figsize=(6, 4))
    for h in histories:
        plt.plot(range(len(h)), h, alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Inertia")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    data_path = Path("CSC2042S-Assignment1-Data/cleaned_data.csv")
    X = load_dataset(data_path)
    k = 3
    runs = 5

    Path("figures").mkdir(exist_ok=True)

    random_exp = run_experiments(X, k, init="random", runs=runs)
    kpp_exp = run_experiments(X, k, init="kmeans++", runs=runs)

    plot_histories(random_exp["histories"], "Random Initialization", "figures/random_init_convergence.png")
    plot_histories(kpp_exp["histories"], "K-means++ Initialization", "figures/kmeanspp_init_convergence.png")

    print("Random initialization:")
    for r in random_exp["results"]:
        print(r)
    print("Average ARI stability:", random_exp["stability"])

    print("\nK-means++ initialization:")
    for r in kpp_exp["results"]:
        print(r)
    print("Average ARI stability:", kpp_exp["stability"])
