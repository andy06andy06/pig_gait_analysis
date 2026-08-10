import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.base import clone

class OversamplingPipeline(Pipeline):
    def fit(self, X, y=None, **fit_params):
        # We need to access fit_params for each step
        # In sklearn, _check_fit_params is helper. We can use a simpler approach
        # that mimics standard fit.
        
        # 1. Fit and transform all steps except the last one
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is None or transform == 'passthrough':
                continue
            Xt = transform.fit_transform(Xt, y)
            
        # 2. Resample Xt and y
        y = np.array(y)
        classes, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        
        new_indices = list(range(len(y)))
        rng = np.random.default_rng(42)
        
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            cls_count = len(cls_indices)
            if cls_count < max_count:
                extra_indices = rng.choice(cls_indices, size=(max_count - cls_count), replace=True)
                new_indices.extend(extra_indices)
                
        new_indices = np.array(new_indices)
        Xt_resampled = Xt[new_indices]
        y_resampled = y[new_indices]
        
        # 3. Fit final estimator on resampled data
        self.steps[-1][1].fit(Xt_resampled, y_resampled)
        return self

# Verify
pipe = OversamplingPipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif, k=2)),
    ('svm', SVC(probability=True))
])

X = np.random.rand(10, 5)
y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])

param_grid = {
    'select__k': [2],
    'svm__C': [1, 10]
}

gs = GridSearchCV(pipe, param_grid, cv=LeaveOneOut())
gs.fit(X, y)
print("GridSearch Best Params:", gs.best_params_)
print("GridSearch Predictions:", gs.predict(X))
