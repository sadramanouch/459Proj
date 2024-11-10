import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

def performClassification(X, y, save_path="classification_results"):
    # Ensure the save_path directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
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
            clf, X_train, y_train, cv=skf,
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            return_train_score=False
        )
        # Fit the classifier on the whole training set
        clf.fit(X_train, y_train)
        # Predict on the test set
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # Binarize labels for multi-class ROC AUC
        y_test_binarized = label_binarize(y_test, classes=np.unique(y))
        n_classes = y_test_binarized.shape[1]

        if y_proba is not None:
            # For classifiers that provide probability estimates
            if n_classes > 2:
                auc = roc_auc_score(y_test_binarized, y_proba, average='macro', multi_class='ovo')
            else:
                auc = roc_auc_score(y_test, y_proba[:, 1])
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
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'classifier': clf
        }

        # Classification Report
        class_report = classification_report(y_test, y_pred, zero_division=0)
        print(f"Classification Report for {name}:\n{class_report}")

        # Confusion Matrix
        plt.figure(figsize=(8,6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
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
                         label='Class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc="lower right")
            plt.savefig(f"{save_path}/roc_curve_{name.replace(' ', '_')}.png")
            plt.close()

    # perform Random Search on Random Forest for hyperparameter tuning
    print("\nPerforming Random Search for Random Forest hyperparameter tuning...")

    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    random_search_rf = RandomizedSearchCV(
        classifiers['Random Forest'], param_distributions=param_grid,
        n_iter=10, scoring='accuracy', cv=skf, random_state=42, n_jobs=-1
    )
    random_search_rf.fit(X_train, y_train)
    tuned_rf = random_search_rf.best_estimator_

    tuned_rf.fit(X_train, y_train)
    y_pred_tuned = tuned_rf.predict(X_test)
    y_proba_tuned = tuned_rf.predict_proba(X_test) if hasattr(tuned_rf, "predict_proba") else None

    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    precision_tuned = precision_score(y_test, y_pred_tuned, average='macro', zero_division=0)
    recall_tuned = recall_score(y_test, y_pred_tuned, average='macro', zero_division=0)
    f1_tuned = f1_score(y_test, y_pred_tuned, average='macro', zero_division=0)

    if y_proba_tuned is not None:
        if n_classes > 2:
            auc_tuned = roc_auc_score(y_test_binarized, y_proba_tuned, average='macro', multi_class='ovo')
        else:
            auc_tuned = roc_auc_score(y_test, y_proba_tuned[:, 1])
    else:
        auc_tuned = None

    results["Random Forest (Tuned)"] = {
        'test_metrics': {
            'accuracy': accuracy_tuned,
            'precision': precision_tuned,
            'recall': recall_tuned,
            'f1_score': f1_tuned,
            'roc_auc': auc_tuned
        },
        'y_test': y_test,
        'y_pred': y_pred_tuned,
        'y_proba': y_proba_tuned,
        'classifier': tuned_rf
    }

    return results