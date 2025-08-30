import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_features(csv_path: Path):
    """Load preprocessed data and return numeric feature matrix and names."""
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError(
            "No numeric columns found. Make sure the LFS data file is fully downloaded."
        )
    return numeric.values, numeric.columns


def interpret_components(pca: PCA, feature_names, top_n=5):
    """Return DataFrame of top contributing features for first three components."""
    loadings = pd.DataFrame(
        pca.components_[:3].T,
        index=feature_names,
        columns=["PC1", "PC2", "PC3"],
    )
    top = loadings.apply(lambda s: s.abs().sort_values(ascending=False).head(top_n))
    top.to_csv("figures/q5_top_loadings.csv")
    return top


def plot_explained_variance(pca: PCA):
    exp = pca.explained_variance_ratio_
    cum = exp.cumsum()
    fig, ax = plt.subplots()
    ax.bar(range(1, len(exp) + 1), exp)
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance")
    fig.tight_layout()
    fig.savefig("figures/q5_pca_explained_variance.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(cum) + 1), cum, marker="o")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Cumulative Explained Variance")
    fig.tight_layout()
    fig.savefig("figures/q5_pca_cumulative_variance.png")
    plt.close(fig)


def plot_tsne(X, labels, path: str):
    tsne = TSNE(n_components=2, init="pca", random_state=0, learning_rate="auto")
    emb = tsne.fit_transform(X)
    fig, ax = plt.subplots()
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=15)
    legend = ax.legend(*sc.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    ax.set_title("t-SNE Visualization")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pca_2d(X, labels, path: str):
    fig, ax = plt.subplots()
    sc = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=15)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend = ax.legend(*sc.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pca_3d(X, labels, path: str):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap="tab10", s=15)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    legend = ax.legend(*sc.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    data_path = Path("CSC2042S-Assignment1-Data/cleaned_data.csv")
    X_full, feature_names = load_features(data_path)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    pca = PCA(n_components=10, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    plot_explained_variance(pca)
    top_loadings = interpret_components(pca, feature_names)
    print("Top PCA loadings:\n", top_loadings)

    k = 8
    kmeans_orig = KMeans(n_clusters=k, random_state=0)
    labels_orig = kmeans_orig.fit_predict(X_scaled)

    kmeans_pca = KMeans(n_clusters=k, random_state=0)
    labels_pca = kmeans_pca.fit_predict(X_pca[:, :3])

    sil_orig = silhouette_score(X_scaled, labels_orig)
    sil_pca = silhouette_score(X_pca[:, :3], labels_pca)
    print(f"Silhouette original: {sil_orig:.3f}\nSilhouette PCA: {sil_pca:.3f}")

    plot_tsne(X_scaled, labels_orig, "figures/q5_tsne_original_labels.png")
    plot_tsne(X_pca[:, :3], labels_pca, "figures/q5_tsne_pca_labels.png")

    plot_pca_2d(X_pca[:, :2], labels_orig, "figures/q5_pca2_original_labels.png")
    plot_pca_2d(X_pca[:, :2], labels_pca, "figures/q5_pca2_pca_labels.png")
    plot_pca_3d(X_pca[:, :3], labels_orig, "figures/q5_pca3_original_labels.png")
    plot_pca_3d(X_pca[:, :3], labels_pca, "figures/q5_pca3_pca_labels.png")


if __name__ == "__main__":
    main()
