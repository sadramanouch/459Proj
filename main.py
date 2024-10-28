from ucimlrepo import fetch_ucirepo 
from preprocessing import preprocessData
from clustering import applyClustering
from eda import performEDA
import pandas as pd

# fetch dataset
wine_quality = fetch_ucirepo(id=186) 

# data (as pandas dataframes) 
feature_headers = wine_quality.data.headers[:-2]

# Data (as pandas DataFrames)
X = pd.DataFrame(wine_quality.data.features, columns=feature_headers)
y = pd.Series(wine_quality.data.targets.squeeze(), name='quality')
performEDA(X, y)

# Preprocess data
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