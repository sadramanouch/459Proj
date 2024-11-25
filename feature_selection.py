import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def mutual_info_selection(X, y, k='all'):
    os.makedirs("feature_selection", exist_ok=True)
    
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    feature_scores = pd.Series(selector.scores_, index=X.columns)
    feature_scores.sort_values(ascending=False, inplace=True)

    print("\nTop Features by Mutual Information:")
    print(feature_scores)

    plt.figure(figsize=(10, 6))
    feature_scores.plot(kind="bar")
    plt.title("Feature Importance based on Mutual Information")
    plt.xlabel("Features")
    plt.ylabel("Mutual Information Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("feature_selection/mutual_info_scores.png")
    plt.close()

    return X.columns[selector.get_support()]
