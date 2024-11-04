from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

def rfe_selection(X, y, model=None, n_features=10):
    if model is None:
        model = RandomForestClassifier()
    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(X, y)
    return X.columns[rfe.support_]

def lasso_selection(X, y, alpha=0.01):
    lasso = LassoCV(cv=5, alphas=[alpha])
    lasso.fit(X, y)
    return X.columns[lasso.coef_ != 0]

def mutual_info_selection(X, y, k=10):
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    return X.columns[selector.get_support()]
