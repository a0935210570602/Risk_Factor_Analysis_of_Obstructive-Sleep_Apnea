# smote_augmenter.py
from imblearn.over_sampling import SMOTE
import numpy as np

def smote(x, y, random_state=42, k_neighbors=1):
    smote_instance = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_train_balanced, y_train_balanced = smote_instance.fit_resample(x, y)

    return X_train_balanced, y_train_balanced
