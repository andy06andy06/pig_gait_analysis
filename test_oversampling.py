import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.svm import SVC

class OversamplingCV:
    def __init__(self, cv, random_state=42):
        self.cv = cv
        self.random_state = random_state
        
    def split(self, X, y, groups=None):
        for train_idx, test_idx in self.cv.split(X, y, groups):
            y_train = y[train_idx]
            classes, counts = np.unique(y_train, return_counts=True)
            max_count = np.max(counts)
            
            new_train_idx = list(train_idx)
            rng = np.random.default_rng(self.random_state)
            
            for cls in classes:
                cls_indices = train_idx[y_train == cls]
                cls_count = len(cls_indices)
                if cls_count < max_count:
                    extra_indices = rng.choice(cls_indices, size=(max_count - cls_count), replace=True)
                    new_train_idx.extend(extra_indices)
                    
            yield np.array(new_train_idx), test_idx
            
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.cv.get_n_splits(X, y, groups)

X = np.random.rand(10, 5)
y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2]) # unbalanced classes

cv = OversamplingCV(LeaveOneOut())
for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    print(f"Fold {i}: train class counts = {np.bincount(y[train_idx])}")
    break
