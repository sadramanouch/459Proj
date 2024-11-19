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

  # Normalize/Standardize Numerical Features
  scaler = StandardScaler()
  X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

  # Data Augmentation using SMOTE
  smote = SMOTE(k_neighbors=2, random_state=42)
  X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

  return {"X": X_resampled, "y": y_resampled}