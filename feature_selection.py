from sklearn.feature_selection import SelectKBest, mutual_info_classif

def mutual_info_selection(X, y, k=10):
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    return X.columns[selector.get_support()]
