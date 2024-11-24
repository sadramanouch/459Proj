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
from sklearn.preprocessing import StandardScaler

def parse_args() -> Dict[str, any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outlier-method", type=str, default="LOF")

    args = parser.parse_args()
    return vars(args)

class Data:
    def __init__(self, features, targets, headers):
        self.features = features
        self.targets = targets
        self.headers = headers

class WineQuality:
    def __init__(self, data):
        self.data = data

def main() -> None:
    args = parse_args()

    # Fetch dataset
    wine_quality = fetch_ucirepo(id=186) 

    # Data (as pandas DataFrames)
    print("Loading Data")
    # Include all headers
    all_headers = wine_quality.data.headers
    # Exclude 'type' and 'quality' columns
    feature_headers = [h for h in all_headers if h not in ['type', 'quality', 'color']]

    # Create DataFrame with all columns
    data_df = pd.DataFrame(wine_quality.data.features, columns=all_headers)
    X_original = data_df[feature_headers]
    y_original = pd.Series(wine_quality.data.targets.squeeze(), name='quality')

    performEDA(X_original, y_original)

    # Split the dataset before preprocessing to avoid data leakage
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_original, y_original, test_size=0.2, random_state=42, stratify=y_original
    )

    # Create Data object for training data
    data_train = Data(
        features=X_train_raw.values, 
        targets=y_train_raw.values, 
        headers=feature_headers
    )
    # Create WineQuality object for training data
    wine_quality_train = WineQuality(data=data_train)

    # Preprocess training data
    print("Preprocessing Training Data")
    preprocessed_train = preprocessData(wine_quality_train)
    X_train = preprocessed_train["X"]
    y_train = preprocessed_train["y"]

    # Feature Selection on Training Data
    selected_features_mi = mutual_info_selection(X_train, y_train)
    X_train_mi = X_train[selected_features_mi]

    # Preprocess test data (without SMOTE)
    print("Preprocessing Test Data")
    # Fit scaler on X_train before SMOTE
    scaler = StandardScaler()
    scaler.fit(X_train_raw[selected_features_mi])
    X_test_selected = X_test_raw[selected_features_mi]
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_selected), columns=selected_features_mi)
    X_test_mi = X_test_scaled
    y_test = y_test_raw.reset_index(drop=True)

    # Outlier Detection on Training Data (Optional)
    print("Outlier Detection")
    outlier_method = args["outlier_method"]
    outliers_removed = outlierDetection(X_train_mi, y_train, outlier_method)
    X_train_mi = outliers_removed["X"]
    y_train = outliers_removed["y"]

    # Tuning Clustering Algorithms
    # ========================================
    # # KMeans
    # n_clusters_range = range(2, 10)
    # best_kmeans_params = tune_kmeans(X, n_clusters_range)
    # print("Best KMeans Parameters:", best_kmeans_params)

    # # Hierarchical Clustering
    # n_clusters_range = range(2, 10)
    # best_hierarchical_params = tune_hierarchical(X, n_clusters_range)
    # print("Best Hierarchical Parameters:", best_hierarchical_params)

    # # DBSCAN
    # eps_values = np.linspace(0.1, 2.0, 10)
    # min_samples_values = range(2, 10)
    # best_dbscan_params = tune_dbscan(X, eps_values, min_samples_values)
    # print("Best DBSCAN Parameters:", best_dbscan_params)
    # ========================================

    # Classification
    print("Performing Classification")
    classification_results = performClassification(
        X_train_mi, y_train, X_test_mi, y_test, save_path="classification_results"
    )

    # Print classification evaluation metrics
    for classifier_name, result in classification_results.items():
        print(f"\n{classifier_name} Test Metrics:")
        for metric_name, metric_value in result['test_metrics'].items():
            print(f"{metric_name}: {metric_value}")

if __name__ == "__main__":
    main()
