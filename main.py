from ucimlrepo import fetch_ucirepo 
from preprocessing import preprocessData
from clustering import applyClustering

# fetch dataset
wine_quality = fetch_ucirepo(id=186) 

# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 

# preprocess data
preprocessedData = preprocessData(wine_quality)
X = preprocessedData["X"]
y = preprocessedData["y"]

# Clustering
clustering_results = applyClustering(X)

print("Clustering Evaluation Metrics:")
for method, metrics in clustering_results.items():
    print(f"\n{method} Results:")
    for metric, score in metrics.items():
        print(f"{metric}: {score}")