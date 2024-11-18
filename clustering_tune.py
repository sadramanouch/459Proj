from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering

# Tuning KMeans
def tune_kmeans(X, n_clusters_values, random_state=42):
    best_params = {"n_clusters": None, "silhouette_score": -1}
    
    for n_clusters in n_clusters_values:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X)
        
        sil_score = silhouette_score(X, labels)
        if sil_score > best_params["silhouette_score"]:
            best_params = {
                "n_clusters": n_clusters,
                "silhouette_score": sil_score,
            }
    
    return best_params

# Tuning Hierarchical Clustering
def tune_hierarchical(X, n_clusters_values, linkage_methods=None):
    if linkage_methods is None:
        linkage_methods = ["ward", "complete", "average", "single"]
    
    best_params = {"n_clusters": None, "linkage": None, "silhouette_score": -1}
    
    for n_clusters in n_clusters_values:
        for linkage in linkage_methods:
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = hierarchical.fit_predict(X)
            
            sil_score = silhouette_score(X, labels)
            if sil_score > best_params["silhouette_score"]:
                best_params = {
                    "n_clusters": n_clusters,
                    "linkage": linkage,
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
            
            # Skip cases where all points are noise (-1) or only one cluster is formed
            if len(set(labels)) > 1:
                sil_score = silhouette_score(X, labels)
                if sil_score > best_params["silhouette_score"]:
                    best_params = {
                        "eps": eps,
                        "min_samples": min_samples,
                        "silhouette_score": sil_score,
                    }
    
    return best_params