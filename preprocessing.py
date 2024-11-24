import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocessData(wine_quality):
    # Exclude 'type', 'quality', and 'color' columns
    feature_headers = [h for h in wine_quality.data.headers if h not in ['type', 'quality', 'color']]
    
    # Extract features and target variable
    X = pd.DataFrame(wine_quality.data.features, columns=wine_quality.data.headers)
    y = pd.Series(wine_quality.data.targets.squeeze(), name='quality')
    
    # Exclude 'type' and 'color' columns from features
    X = X[feature_headers]
    
    # Ensure all features are numeric
    non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric_cols:
        print(f"Non-numeric columns found: {non_numeric_cols}")
        X = X.drop(columns=non_numeric_cols)
    
    # Remove zero variance features
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0)
    selector.fit(X)
    constant_columns = [column for column in X.columns if column not in X.columns[selector.get_support()]]
    if constant_columns:
        print(f"Zero variance features found: {constant_columns}")
        X = X.drop(columns=constant_columns)
    
    # Reset index
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    
    # Normalize/Standardize Numerical Features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Data Augmentation using SMOTE
    from collections import Counter
    class_counts = Counter(y)
    min_class_size = min(class_counts.values())
    k_neighbors = min(min_class_size - 1, 5)
    if k_neighbors < 1:
        k_neighbors = 1
    print(f"Using k_neighbors={k_neighbors} for SMOTE")
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    return {"X": X_resampled, "y": y_resampled}
