import pandas as pd
from sklearn import model_selection
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def feature_selection(x, y, config):
    """
    根據 config 配置，使用不同的估計器進行特徵選取，並回傳選取後的特徵名稱清單。

    config 範例:
    {
        "method": "logistic",   # 可選 "logistic"、"linear" 或 "knn"
        "n_features_to_select": 'auto',  # 選取特徵數目，預設 'auto'
        "direction": "backward",         # 選取方向，'backward' 或 'forward'
        # 若 method 為 "logistic"，可傳入：
        "max_iter": 1000,
        # 若 method 為 "knn"，可傳入：
        "n_neighbors": 3
    }

    Args:
        x (DataFrame): 特徵資料，需帶欄位名稱。
        y (array-like): 標籤資料。
        config (dict): 特徵選取配置字典。

    Returns:
        list: 選取後的特徵名稱清單。
    """
    method = config.get("method", "linear")
    n_features_to_select = config.get("n_features_to_select", "auto")
    direction = config.get("direction", "backward")

    if method == "logistic":
        max_iter = config.get("max_iter", 1000)
        estimator = LogisticRegression(max_iter=max_iter)
    elif method == "linear":
        estimator = LinearRegression()
    elif method == "knn":
        n_neighbors = config.get("n_neighbors", 3)
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        raise ValueError(f"Unsupported method: {method}")

    sfs = SequentialFeatureSelector(estimator, 
                                    n_features_to_select=n_features_to_select, 
                                    direction=direction)
    sfs.fit(x, y)
    feature_mask = sfs.get_support()
    selected_features = x.columns[feature_mask].tolist()
    return selected_features
