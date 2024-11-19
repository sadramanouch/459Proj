from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def applyClustering(X, save_path=None):
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=4, n_init=10,random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=4)
    hierarchical_labels = hierarchical.fit_predict(X)
    
    # DBSCAN Clustering
    dbscan = DBSCAN(eps=2, min_samples=4)
    dbscan_labels = dbscan.fit_predict(X)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Plot clustering results
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='viridis', s=50)
    plt.title("K-Means Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=hierarchical_labels, palette='viridis', s=50)
    plt.title("Hierarchical Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dbscan_labels, palette='viridis', s=50)
    plt.title("DBSCAN Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    if save_path:
        plt.savefig(os.path.join(save_path, "clustering.png"))
    
    plt.close()
    
    # Evaluate clustering performance
    evaluation_metrics = {
        "KMeans": {
            "Silhouette Score": silhouette_score(X, kmeans_labels),
            "Calinski-Harabasz Index": calinski_harabasz_score(X, kmeans_labels),
            "Davies-Bouldin Index": davies_bouldin_score(X, kmeans_labels),
        },
        "Hierarchical": {
            "Silhouette Score": silhouette_score(X, hierarchical_labels) if len(set(hierarchical_labels)) > 1 else "N/A",
            "Calinski-Harabasz Index": calinski_harabasz_score(X, hierarchical_labels) if len(set(hierarchical_labels)) > 1 else "N/A",
            "Davies-Bouldin Index": davies_bouldin_score(X, hierarchical_labels) if len(set(hierarchical_labels)) > 1 else "N/A",
        },
        "DBSCAN": {
            "Silhouette Score": silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else "N/A",
            "Calinski-Harabasz Index": calinski_harabasz_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else "N/A",
            "Davies-Bouldin Index": davies_bouldin_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else "N/A",
        },
    }
    
    return evaluation_metrics
