from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

def applyClustering(X):
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Plot K-Means clustering result
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='viridis', s=50)
    plt.title("K-Means Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    # Plot DBSCAN clustering result
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dbscan_labels, palette='viridis', s=50)
    plt.title("DBSCAN Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate clustering performance
    evaluation_metrics = {
        "KMeans": {
            "Silhouette Score": silhouette_score(X, kmeans_labels),
            "Calinski-Harabasz Index": calinski_harabasz_score(X, kmeans_labels),
            "Davies-Bouldin Index": davies_bouldin_score(X, kmeans_labels),
        },
        "DBSCAN": {
            "Silhouette Score": silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else "N/A",
            "Calinski-Harabasz Index": calinski_harabasz_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else "N/A",
            "Davies-Bouldin Index": davies_bouldin_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else "N/A",
        },
    }
    
    return evaluation_metrics
