import numpy as np
import pandas as pd
import os
from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

def performClassification(X_train, y_train, X_test, y_test, save_path="classification_results"):
    # Ensure the save_path directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Convert y to categorical if it's not already
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Group wine quality scores into categories (e.g., Low, Medium, High)
    bins = [0, 5, 6, 10]
    labels = ['Low', 'Medium', 'High']
    y_train = pd.cut(y_train, bins=bins, labels=labels, include_lowest=True)
    y_test = pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'k-NN': KNeighborsClassifier()
    }

    # Initialize results dictionary
    results = {}

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in classifiers.items():
        print(f"\nTraining and evaluating {name}...")
        # Cross-validation
        cv_results = cross_validate(
            clf, X_train, y_train_encoded, cv=skf,
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            return_train_score=False
        )
        # Fit the classifier on the whole training set
        clf.fit(X_train, y_train_encoded)
        # Predict on the test set
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)

        # Binarize labels for multi-class ROC AUC
        y_test_binarized = label_binarize(y_test_encoded, classes=np.unique(y_train_encoded))
        n_classes = y_test_binarized.shape[1]

        if y_proba is not None:
            # For classifiers that provide probability estimates
            auc = roc_auc_score(y_test_binarized, y_proba, average='macro', multi_class='ovo')
        else:
            auc = None

        # Save results
        results[name] = {
            'cv_results': cv_results,
            'test_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': auc
            },
            'y_test': y_test_encoded,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'classifier': clf
        }

        # Classification Report
        class_report = classification_report(y_test_encoded, y_pred, target_names=le.classes_, zero_division=0)
        print(f"Classification Report for {name}:\n{class_report}")

        # Confusion Matrix
        plt.figure(figsize=(8,6))
        cm = confusion_matrix(y_test_encoded, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f"{save_path}/confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close()

        # ROC Curve for multi-class classification
        if y_proba is not None:
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
                roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_proba[:, i])

            # Plot all ROC curves
            plt.figure()
            colors = plt.cm.get_cmap('tab10', n_classes)
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], lw=2, color=colors(i),
                         label='Class {0} (AUC = {1:0.2f})'.format(le.classes_[i], roc_auc[i]))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc="lower right")
            plt.savefig(f"{save_path}/roc_curve_{name.replace(' ', '_')}.png")
            plt.close()

    return results
