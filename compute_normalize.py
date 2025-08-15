import numpy as np
import math, copy
from sklearn.preprocessing import StandardScaler

def zscore_normalize_feature(X):
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    return X_norm