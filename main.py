from ucimlrepo import fetch_ucirepo
from preprocessing import preprocessData
from clustering import applyClustering
from eda import performEDA
from outlier import outlierDetection
import pandas as pd
from feature_selection import mutual_info_selection

# fetch dataset
wine_quality = fetch_ucirepo(id=186) 

# data (as pandas dataframes) 
feature_headers = wine_quality.data.headers[:-2]

# Data (as pandas DataFrames)
print("Performing EDA")
X = pd.DataFrame(wine_quality.data.features, columns=feature_headers)
y = pd.Series(wine_quality.data.targets.squeeze(), name='quality')
performEDA(X, y)

# Preprocess data
print("Preprocessing Data")
preprocessedData = preprocessData(wine_quality)
X = preprocessedData["X"]
y = preprocessedData["y"]

# Outlier Detection
# print("Outlier Detection")
# X = outlierDetection(X)

# Clustering with Mutual Information-selected features
selected_features_mi = mutual_info_selection(X, y)
X_mi = X[selected_features_mi]

print("Clustering with MI-selected features")
clustering_results_mi = applyClustering(X_mi, save_path="clustering_plots")

print("Clustering Evaluation Metrics with MI-selected features:")
for clustering_method, metrics in clustering_results_mi.items():
    print(f"\n{clustering_method} Results:")
    for metric, score in metrics.items():
        print(f"{metric}: {score}")
