import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE

def remove_correlated_features(X_train, X_test, threshold=0.9):
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_train_reduced = X_train.drop(columns=to_drop)
    X_test_reduced = X_test.drop(columns=to_drop)
    return X_train_reduced, X_test_reduced, to_drop

def smote(X,y):
    smote = SMOTE(random_state=21)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def tune_logistic_regression(X, y):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [100, 200, 300],
    }
    lr = LogisticRegression(random_state=21)
    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

def tune_random_forest(X, y):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    }
    rf = RandomForestClassifier(random_state=21)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

def tune_svm(X, y):
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1]
    }
    svm = SVC(kernel='rbf', random_state=21, probability=True)
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

def tune_xgb(X, y):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb_clf = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

def tune_voting(X,y, base_models):
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', base_models['lr']),
            ('rf', base_models['rf']),
            ('svm', base_models['svm']),
            ('xgb', base_models['xgb'])
        ],
        voting='hard',
        weights=[3, 10, 3, 8] 
    )
    voting_clf.fit(X, y)
    return voting_clf