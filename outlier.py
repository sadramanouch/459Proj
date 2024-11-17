from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import os

def outlierDetection(X, y, outlier_method):
    if outlier_method == "LOF":
        return outlierDetectionLOF(X, y, outlier_method)
    elif outlier_method == "IF":
        return outlierDetectionIsolationForest(X, y, outlier_method)
    elif outlier_method == "EE":
        return outlierDetectionEllipticEnvelope(X, y, outlier_method)

def outlierDetectionLOF(X, y, outlier_method):
    clf = LocalOutlierFactor(n_neighbors=20, contamination='auto')
    outlier_labels = clf.fit_predict(X)

    # Reduce dimensionality to 2 for plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    X_pca["outliers"] = outlier_labels

    plotOutliers(X_pca, outlier_method)
    removed_outliers = removeOutliers(X, y, outlier_labels)
    return {"X": removed_outliers["X"], "y": removed_outliers["y"]}

def outlierDetectionIsolationForest(X, y, outlier_method):
    clf = IsolationForest(random_state=0)
    outlier_labels = clf.fit_predict(X)

    # Reduce dimensionality to 2 for plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    X_pca["outliers"] = outlier_labels

    plotOutliers(X_pca, outlier_method)
    removed_outliers = removeOutliers(X, y, outlier_labels)
    return {"X": removed_outliers["X"], "y": removed_outliers["y"]}

def outlierDetectionEllipticEnvelope(X, y, outlier_method):
    clf = EllipticEnvelope(random_state=0, contamination=0.1) 
    clf.fit(X)
    outlier_labels = clf.predict(X)

    # Reduce dimensionality to 2 for plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    X_pca["outliers"] = outlier_labels

    plotOutliers(X_pca, outlier_method)
    removed_outliers = removeOutliers(X, y, outlier_labels)
    return {"X": removed_outliers["X"], "y": removed_outliers["y"]}

def plotOutliers(X, outlier_method):
    X_inliers = X[X["outliers"] == 1]
    X_outliers = X[X["outliers"] == -1]

    plt.figure(figsize=(10, 6))
    plt.scatter(X_inliers["PC1"], X_inliers["PC2"], label="Inliers", color="blue", s=10)
    plt.scatter(X_outliers["PC1"], X_outliers["PC2"], label="Outliers", color="red", s=10)

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.title(f"Outlier Detection using {outlier_method}")

    # Create a directory to save plots
    plots_dir = "outlier_plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, outlier_method + ".png"))
    plt.close()


def removeOutliers(X, y, outlier_labels):
    inliers_mask = outlier_labels == 1
    X_cleaned = X[inliers_mask]
    y_cleaned = y[inliers_mask]
    return {"X": X_cleaned, "y": y_cleaned}
