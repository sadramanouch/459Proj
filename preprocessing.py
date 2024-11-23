import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocessData(wine_quality):
  feature_headers = wine_quality.data.headers[:-2]

  # Extract features and target variable
  X = pd.DataFrame(wine_quality.data.features, columns=feature_headers)
  y = pd.Series(wine_quality.data.targets.squeeze(), name='quality')

  # Reset index
  X.reset_index(drop=True, inplace=True)
  y.reset_index(drop=True, inplace=True)

  return {"X": X, "y": y}