from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import os

def applyClustering(X, save_path=None):
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    X_pca = pca.fit_transform(X)
    X_tsne = tsne.fit_transform(X)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=4)
    hierarchical_labels = hierarchical.fit_predict(X)

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=2, min_samples=4)
    dbscan_labels = dbscan.fit_predict(X)

    # Plot clustering results
    # K-Means: Visualize using PCA
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 3, 1)
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette="viridis", s=50
    )
    plt.title("K-Means Clustering (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Hierarchical Clustering: Visualize with Dendrogram
    plt.subplot(1, 3, 2)
    linked = linkage(X, method="ward")
    dendrogram(linked, truncate_mode="lastp", p=10, show_leaf_counts=True)
    plt.title("Hierarchical Clustering (Dendrogram)")
    plt.xlabel("Cluster")
    plt.ylabel("Distance")

    # DBSCAN: Visualize using t-SNE
    plt.subplot(1, 3, 3)
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1], hue=dbscan_labels, palette="viridis", s=50
    )
    plt.title("DBSCAN Clustering (t-SNE)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    if save_path:
        plt.savefig(os.path.join(save_path, "clustering.png"))
    plt.close()

    # Evaluate clustering performance
    evaluation_metrics = {
        "KMeans": {
            "Silhouette Score": silhouette_score(X, kmeans_labels),
            "Calinski-Harabasz Index": calinski_harabasz_score(X, kmeans_labels),
        },
        "Hierarchical": {
            "Silhouette Score": silhouette_score(X, hierarchical_labels)
            if len(set(hierarchical_labels)) > 1
            else "N/A",
            "Calinski-Harabasz Index": calinski_harabasz_score(X, hierarchical_labels)
            if len(set(hierarchical_labels)) > 1
            else "N/A",
        },
        "DBSCAN": {
            "Davies-Bouldin Index": davies_bouldin_score(X, dbscan_labels)
            if len(set(dbscan_labels)) > 1
            else "N/A",
        },
    }

    return evaluation_metrics
