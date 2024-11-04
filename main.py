from ucimlrepo import fetch_ucirepo
from preprocessing import preprocessData
from clustering import applyClustering
from eda import performEDA
from outlier import outlierDetection
import pandas as pd
from feature_selection import rfe_selection, lasso_selection, mutual_info_selection

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

# Clustering without feature selection
print("Clustering without Feature Selection")
clustering_results = applyClustering(X)
print("Clustering Evaluation Metrics (Without Feature Selection):")
for method, metrics in clustering_results.items():
    print(f"\n{method} Results:")
    for metric, score in metrics.items():
        print(f"{metric}: {score}")

# Feature selection methods
# 1. Recursive Feature Elimination (RFE)
print("Feature Selection - RFE")
selected_features_rfe = rfe_selection(X, y)
X_rfe = X[selected_features_rfe]
print("Clustering with RFE-selected features")
clustering_results_rfe = applyClustering(X_rfe)

# 2. Lasso Regression
print("Feature Selection - Lasso")
selected_features_lasso = lasso_selection(X, y)
X_lasso = X[selected_features_lasso]
print("Clustering with Lasso-selected features")
clustering_results_lasso = applyClustering(X_lasso)

# 3. Mutual Information
print("Feature Selection - Mutual Information")
selected_features_mi = mutual_info_selection(X, y)
X_mi = X[selected_features_mi]
print("Clustering with MI-selected features")
clustering_results_mi = applyClustering(X_mi)

# Outlier Detection
print("Outlier Detection")
X = outlierDetection(X)

# Compare clustering performance with feature selection
print("Clustering Evaluation with Feature Selection:")
for method, results in zip(
    ["RFE", "Lasso", "Mutual Information"], 
    [clustering_results_rfe, clustering_results_lasso, clustering_results_mi]
):
    print(f"\n{method} Results:")
    for clustering_method, metrics in results.items():
        print(f"\n{clustering_method}:")
        for metric, score in metrics.items():
            print(f"{metric}: {score}")
