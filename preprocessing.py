import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocessData(wine_quality):
  X = wine_quality.data.features 
  y = wine_quality.data.targets 

  # remove all the rows that have 9 as the class label
  X = pd.DataFrame(X, columns=wine_quality.data.headers[:-2])
  y = y.squeeze()
  X = X[y != 9]
  y = y[y != 9]

  # normalize/standardize numerical features
  scaler = StandardScaler()
  scaled_features = scaler.fit_transform(X)
  X = pd.DataFrame(scaled_features, columns=wine_quality.data.headers[:-2])

  # data augmentation using SMOTE
  smote = SMOTE(k_neighbors=2)
  X_resampled, y_resampled = smote.fit_resample(X, y)

  return {"X": X_resampled, "y": y_resampled}