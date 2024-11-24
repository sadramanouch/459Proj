from ucimlrepo import fetch_ucirepo
from preprocessing import preprocessData
from clustering import applyClustering
from eda import performEDA
from outlier import outlierDetection
import pandas as pd
from feature_selection import mutual_info_selection
from classification import performClassification
from typing import Dict
import argparse
from clustering_tune import tune_kmeans, tune_hierarchical, tune_dbscan
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def parse_args() -> Dict[str, any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outlier-method", type=str, default="LOF")

    args = parser.parse_args()
    return vars(args)

def main() -> None:
    args = parse_args()

    # fetch dataset
    wine_quality = fetch_ucirepo(id=186) 

    # data (as pandas dataframes) 
    feature_headers = wine_quality.data.headers[:-2]

    # Data (as pandas DataFrames)
    print("Performing EDA")
    X = pd.DataFrame(wine_quality.data.features, columns=feature_headers)
    y = pd.Series(wine_quality.data.targets.squeeze(), name='quality')
    performEDA(X, y)

    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess data
    print("Preprocessing Data")
    preprocessedData = preprocessData(X, X_train, X_test, y_train)
    X_train_resampled = preprocessedData["X_train_resampled"]
    y_train_resampled = preprocessedData["y_train_resampled"]
    X_test = preprocessedData["X_test"]

    # Outlier Detection
    print("Outlier Detection")
    outlier_method = args["outlier_method"]
    outliers_removed = outlierDetection(X_train_resampled, y_train_resampled, outlier_method)
    X = outliers_removed["X"]
    y = outliers_removed["y"]
    print(X)

    # Clustering with Mutual Information-selected features
    selected_features_mi = mutual_info_selection(X, y)
    X_mi = X[selected_features_mi]
    X_original_scaled_mi = X_test[selected_features_mi]

    # # Tuning Clustering Algorithms
    # # ========================================
    # # # KMeans
    # # n_clusters_range = range(2, 10)
    # # best_kmeans_params = tune_kmeans(X, n_clusters_range)
    # # print("Best KMeans Parameters:", best_kmeans_params)

    # # # Hierarchical Clustering 
    # # n_clusters_range = range(2, 10)
    # # best_hierarchical_params = tune_hierarchical(X, n_clusters_range)
    # # print("Best Hierarchical Parameters:", best_hierarchical_params)

    # # # DBSCAN
    # # eps_values = np.linspace(0.1, 2.0, 10)
    # # min_samples_values = range(2, 10)
    # # best_dbscan_params = tune_dbscan(X, eps_values, min_samples_values)
    # # print("Best DBSCAN Parameters:", best_dbscan_params)
    # # ========================================

    print("Clustering with MI-selected features")
    clustering_results_mi = applyClustering(X_mi, save_path="clustering_plots")

    print("Clustering Evaluation Metrics with MI-selected features:")
    for clustering_method, metrics in clustering_results_mi.items():
        print(f"\n{clustering_method} Results:")
        for metric, score in metrics.items():
            print(f"{metric}: {score}")

    # Classification
    print("Performing Classification")
    classification_results = performClassification(X_mi, y, X_original_scaled_mi, y_test, save_path="classification_results")

    # Print classification evaluation metrics
    for classifier_name, result in classification_results.items():
        print(f"\n{classifier_name} Test Metrics:")
        for metric_name, metric_value in result['test_metrics'].items():
            print(f"{metric_name}: {metric_value}")

if __name__ == "__main__":
    main()