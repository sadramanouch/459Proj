from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def applyClustering(X, outlier_method, save_path=None):
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    kmeans_clusters = None
    hierarchical_clusters = None
    eps = None
    min_samples = 2

    if outlier_method == "LOF":
        kmeans_clusters = 2
        hierarchical_clusters = 2
        eps = 2
    elif outlier_method == "IF":
        kmeans_clusters = 3
        hierarchical_clusters = 3
        eps = 0.1
    elif outlier_method == "EE":
        kmeans_clusters = 3
        hierarchical_clusters = 2
        eps = 0.1

    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # t-SNE for nonlinear visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=hierarchical_clusters)
    hierarchical_labels = hierarchical.fit_predict(X)

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X)

    # Plot clustering results
    plt.figure(figsize=(20, 6))

    # K-Means: Visualize using PCA
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette="viridis", s=50)
    plt.title("K-Means Clustering (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Hierarchical Clustering: Visualize using PCA
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=hierarchical_labels, palette="viridis", s=50)
    plt.title("Hierarchical Clustering (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # DBSCAN: Visualize using t-SNE
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=dbscan_labels, palette="viridis", s=50)
    plt.title("DBSCAN Clustering (t-SNE)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    if save_path:
        plt.savefig(os.path.join(save_path, f"clustering_{outlier_method}.png"))
    plt.close()

    # Evaluate clustering performance
    evaluation_metrics = {
        "KMeans": {
            "Silhouette Score": silhouette_score(X, kmeans_labels),
        },
        "Hierarchical": {
            "Calinski-Harabasz Index": calinski_harabasz_score(X, hierarchical_labels),
        },
        "DBSCAN": {
            "Davies-Bouldin Index": davies_bouldin_score(X, dbscan_labels)
            if len(set(dbscan_labels)) > 1
            else "N/A",
        },
    }

    return evaluation_metrics
