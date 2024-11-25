import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def mutual_info_selection(X, y, k=10):
    os.makedirs("feature_selection", exist_ok=True)
    
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    feature_scores = pd.Series(selector.scores_, index=X.columns)
    feature_scores.sort_values(ascending=False, inplace=True)

    print("\nTop 10 Features by Mutual Information:")
    print(feature_scores.head(k))

    plt.figure(figsize=(12, 6))
    feature_scores.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Top 10 Features: Importance Based on Mutual Information")
    plt.xlabel("Features")
    plt.ylabel("Mutual Information Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig("feature_selection/mutual_info_scores_top_10.png")
    plt.close()

    return X.columns[selector.get_support()]
