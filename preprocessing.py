import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocessData(X, X_train, X_test, y_train):

  # Normalize/Standardize Numerical Features
  scaler = StandardScaler()
  X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
  X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X.columns)

  # Data Augmentation using SMOTE
  smote = SMOTE(k_neighbors=2, random_state=42)
  X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

  return {"X_train_resampled": X_train_resampled, "y_train_resampled": y_train_resampled, "X_test": X_test}