from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering

# Tuning KMeans
def tune_kmeans(X, n_clusters_values):
    best_params = {"n_clusters": None, "n_init": None, "silhouette_score": -1}
    
    for n_clusters in n_clusters_values:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        
        sil_score = silhouette_score(X, labels)
        if sil_score > best_params["silhouette_score"]:
            best_params = {
                "n_clusters": n_clusters,
                "silhouette_score": sil_score,
            }
    
    return best_params


# Tuning Hierarchical Clustering
def tune_hierarchical(X, n_clusters_values):
    best_params = {"n_clusters": None, "silhouette_score": -1}
    
    for n_clusters in n_clusters_values:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(X)
        
        sil_score = silhouette_score(X, labels)
        if sil_score > best_params["silhouette_score"]:
            best_params = {
                "n_clusters": n_clusters,
                "silhouette_score": sil_score,
            }
    
    return best_params

# Tuning DBSCAN
def tune_dbscan(X, eps_values, min_samples_values):
    best_params = {"eps": None, "min_samples": None, "silhouette_score": -1}
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            if len(set(labels)) > 1:
                sil_score = silhouette_score(X, labels)
                if sil_score > best_params["silhouette_score"]:
                    best_params = {
                        "eps": eps,
                        "min_samples": min_samples,
                        "silhouette_score": sil_score,
                    }
    
    return best_params