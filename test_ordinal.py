import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, LeaveOneOut

class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf=None):
        self.clf = clf
        self.clfs = []
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.clfs = []
        n_classes = len(self.classes_)
        for i in range(n_classes - 1):
            binary_y = (y > i).astype(int)
            clf = clone(self.clf)
            clf.fit(X, binary_y)
            self.clfs.append(clf)
        return self

    def predict_proba(self, X):
        probs = []
        for clf in self.clfs:
            probs.append(clf.predict_proba(X)[:, 1])
        probs = np.array(probs).T
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        class_probs = np.zeros((n_samples, n_classes))
        
        class_probs[:, 0] = 1.0 - probs[:, 0]
        for i in range(1, n_classes - 1):
            class_probs[:, i] = probs[:, i-1] - probs[:, i]
        class_probs[:, -1] = probs[:, -1]
        
        class_probs = np.clip(class_probs, 0.0, 1.0)
        row_sums = class_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        class_probs = class_probs / row_sums
        
        return class_probs

    def predict(self, X):
        class_probs = self.predict_proba(X)
        return np.argmax(class_probs, axis=1)

# Verify with pipeline and grid search
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif, k=2)),
    ('svm', OrdinalClassifier(SVC(probability=True)))
])

param_grid = {
    'select__k': [2],
    'svm__clf__C': [1, 10],
    'svm__clf__kernel': ['linear', 'rbf']
}

X = np.random.rand(10, 5)
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

gs = GridSearchCV(pipe, param_grid, cv=LeaveOneOut())
gs.fit(X, y)
print("Best parameters:", gs.best_params_)
print("Best score:", gs.best_score_)
print("Predictions:", gs.predict(X))
