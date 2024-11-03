from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

# Local Outlier Factor Used since dimensionality of the dataset is not very large
def outlierDetection(X):
  clf = LocalOutlierFactor(n_neighbors=20)
  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(X)
  outlier_labels = clf.fit_predict(X_pca)

  X_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
  X_pca["outliers"] = outlier_labels

  plotOutliers(X_pca)

  X_removed_outliers = removeOutliers(X, outlier_labels)
  return X_removed_outliers

def plotOutliers(X):
  X_inliers = X[X["outliers"] == 1]
  X_outliers = X[X["outliers"] == -1]

  plt.figure(figsize=(10, 6))
  plt.scatter(X_inliers["PC1"], X_inliers["PC2"], label="Inliers", color="blue", s=10)
  plt.scatter(X_outliers["PC1"], X_outliers["PC2"], label="Outliers", color="red", s=10)

  plt.xlabel("Component 1")
  plt.ylabel("Component 2")
  plt.legend()
  plt.title("Outlier Detection using Local Outlier Factor")
  plt.show()

def removeOutliers(X, outlier_labels):
  inliers_mask = outlier_labels == 1
  X_cleaned = X[inliers_mask]
  return X_cleaned