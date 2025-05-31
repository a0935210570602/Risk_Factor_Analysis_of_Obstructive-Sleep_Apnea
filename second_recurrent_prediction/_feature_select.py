import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector, VarianceThreshold, RFE, SelectKBest, chi2, SelectFromModel, RFECV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import xgboost as xgb
from boruta import BorutaPy

# 1. Sequential Feature Selection (使用 SequentialFeatureSelector)
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: .*")

def apply_feature_selection(self):
    print("Feature Selection: ", self.SELECTED_FEATURE_LIST)
    print(self.train_X.shape, self.train_Y.shape)
    self.train_X = self.train_X[self.SELECTED_FEATURE_LIST]
    self.valid_X = self.valid_X[self.SELECTED_FEATURE_LIST]

    print(self.valid_X.shape, self.valid_Y.shape)
    
def sfs_logistic_feature_selection(self):
    config = self.feature_selection_config
    self.feature_selection_config["name"] = "sfs_logistic_feature_selection"
    direction = config.get("direction", "backward")
    n_features_to_select = config.get("n_features_to_select", "auto")
    max_iter = config.get("max_iter", 1000)
    random_state = config.get("random_state", 42)
    
    estimator = LogisticRegression(max_iter=max_iter, random_state=random_state)
    sfs = SequentialFeatureSelector(estimator,
                                    n_features_to_select=n_features_to_select,
                                    direction=direction,
                                    n_jobs=-1)
    
    sfs.fit(self.train_X, self.train_Y)
    self.SELECTED_FEATURE_LIST = self.train_X.columns[sfs.get_support()].tolist()
    apply_feature_selection(self)

def sfs_linear_feature_selection(self):
    config = self.feature_selection_config
    self.feature_selection_config["name"] = "sfs_linear_feature_selection"

    direction = config.get("direction", "backward")
    n_features_to_select = config.get("n_features_to_select", "auto")
    
    estimator = LinearRegression()
    sfs = SequentialFeatureSelector(estimator,
                                    n_features_to_select=n_features_to_select,
                                    direction=direction,
                                    n_jobs=-1)
    sfs.fit(self.train_X, self.train_Y)
    self.SELECTED_FEATURE_LIST = self.train_X.columns[sfs.get_support()].tolist()
    apply_feature_selection(self)

def sfs_knn_feature_selection(self):

    config = self.feature_selection_config
    self.feature_selection_config["name"] = "sfs_knn_feature_selection"
    direction = config.get("direction", "backward")

    n_features_to_select = config.get("n_features_to_select", "auto")
    n_neighbors = config.get("n_neighbors", )
    
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    sfs = SequentialFeatureSelector(estimator,
                                    n_features_to_select=n_features_to_select,
                                    direction=direction,
                                    n_jobs=-1)
    
    sfs.fit(self.train_X, self.train_Y)
    self.SELECTED_FEATURE_LIST = self.train_X.columns[sfs.get_support()].tolist()
    apply_feature_selection(self)

def sfs_random_forest_feature_selection(self):
    print("🚀 Running SFS with Random Forest Feature Selection...")

    # 1. 讀取設定
    config = self.feature_selection_config
    self.feature_selection_config["name"] = "sfs_random_forest_feature_selection"
    
    # 2. 參數空間 (可從 config 裡讀，也可直接在這裡改)
    directions = config.get("grid_directions", [ "backward"])
    n_features_list = config.get("grid_n_features_to_select", [8])
    n_estimators_list = config.get("grid_n_estimators", [100])
    cv = config.get("cv", 5)
    
    X, y = self.train_X, self.train_Y
    best_score = -np.inf
    best_params = {}
    print(9999999)
    # 3. Grid Search
    for direction in directions:
        for n_feat in n_features_list:
            for n_est in n_estimators_list:
                # 建立 SFS + RF
                estimator = RandomForestClassifier(n_estimators=n_est,
                                                   random_state=config.get("random_state", 42))
                sfs = SequentialFeatureSelector(
                    estimator,
                    n_features_to_select=n_feat,
                    direction=direction,
                    scoring="f1",
                    cv=cv,
                    n_jobs=1
                )
                # 先挑特徵
                sfs.fit(X, y)
                sel_idx = sfs.get_support()
                X_sel = X.iloc[:, sel_idx]
                
                # 再用交叉驗證評估選完後的模型
                score = np.mean(cross_val_score(
                    estimator, X_sel, y,
                    scoring="f1", cv=cv, n_jobs=-1
                ))
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        "direction": direction,
                        "n_features_to_select": n_feat,
                        "n_estimators": n_est
                    }
                print(f"[Grid] dir={direction}  n_feat={n_feat}  n_est={n_est}  →  F1={score:.4f}")
    
    # 4. 用最佳參數重訓一遍，並記錄最終選出的特徵
    print(f"\n✅ Best Params: {best_params}  F1={best_score:.4f}")
    estimator = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        random_state=config.get("random_state", 42)
    )
    sfs = SequentialFeatureSelector(
        estimator,
        n_features_to_select=best_params["n_features_to_select"],
        direction=best_params["direction"],
        scoring="f1",
        cv=cv,
        n_jobs=1
    )
    sfs.fit(X, y)
    
    # 5. 存下選到的欄位，並呼叫 apply_feature_selection
    self.SELECTED_FEATURE_LIST = X.columns[sfs.get_support()].tolist()
    print(f"[SFS] Final selected features: {self.SELECTED_FEATURE_LIST}")
    apply_feature_selection(self)
    
# def sfs_random_forest_feature_selection(self):
#     self.train_X = pd.get_dummies(self.train_X, columns=self.categorical_features, drop_first=True)
#     self.valid_X = pd.get_dummies(self.valid_X, columns=self.categorical_features, drop_first=True)
#     self.data_X = pd.get_dummies(self.data_X, columns=self.categorical_features, drop_first=True)

#     config = self.feature_selection_config
#     self.feature_selection_config["name"] = "sfs_random_forest_feature_selection"

#     direction = config.get("direction", "backward")
#     n_features_to_select = config.get("n_features_to_select", "auto")
#     n_estimators = config.get("n_estimators", 200)
#     random_state = config.get("random_state", 42)
    
#     estimator = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
#     sfs = SequentialFeatureSelector(estimator,
#                                     n_features_to_select=n_features_to_select,
#                                     direction=direction,
#                                     n_jobs=-1,
#                                     scoring='1',
#                                     cv=5)
    
#     sfs.fit(self.train_X, self.train_Y)
#     self.SELECTED_FEATURE_LIST = self.train_X.columns[sfs.get_support()].tolist()
#     apply_feature_selection(self)

def sfs_xgboost_feature_selection(self):
    config = self.feature_selection_config
    self.feature_selection_config["name"] = "sfs_xgboost_feature_selection"

    direction = config.get("direction", "backward")
    n_features_to_select = config.get("n_features_to_select", "auto")
    max_depth = config.get("max_depth", 5)
    learning_rate = config.get("learning_rate", 0.1)
    random_state = config.get("random_state", 42)
    
    estimator = xgb.XGBClassifier(max_depth=max_depth,
                                  learning_rate=learning_rate,
                                  use_label_encoder=False,
                                  eval_metric='logloss',
                                  random_state=random_state)
    
    sfs = SequentialFeatureSelector(estimator,
                                    n_features_to_select=n_features_to_select,
                                    direction=direction,
                                    n_jobs=-1)
    
    sfs.fit(self.train_X, self.train_Y)
    self.SELECTED_FEATURE_LIST = self.train_X.columns[sfs.get_support()].tolist()
    apply_feature_selection(self)

def permutation_feature_selection(self):
    config = self.feature_selection_config
    self.feature_selection_config["name"] = "permutation_feature_selection"

    n_estimators = config.get("n_estimators", 100)
    random_state = config.get("random_state", 42)
    n_repeats = config.get("n_repeats", 10)

    estimator = RandomForestClassifier(n_estimators=n_estimators,
                                       random_state=random_state)
    estimator.fit(self.train_X, self.train_Y)
    result = permutation_importance(estimator, self.train_X, self.train_Y,
                                    n_repeats=n_repeats,
                                    random_state=random_state,
                                    n_jobs=-1)
    importance_df = pd.DataFrame({
        'feature': self.train_X.columns,
        'importance': result.importances_mean
    })
    top_k = config.get("top_k", 10)
    selected_features = importance_df.sort_values('importance', ascending=False)\
                                     .head(top_k)["feature"].tolist()
    
    self.SELECTED_FEATURE_LIST = selected_features
    apply_feature_selection(self)

# 3. Boruta Feature Selection
def boruta_feature_selection(self):
    config = self.feature_selection_config
    self.feature_selection_config["name"] = "boruta_feature_selection"

    n_estimators = config.get("n_estimators", 100)
    random_state = config.get("random_state", 42)
    estimator = RandomForestClassifier(n_estimators=n_estimators,
                                       random_state=random_state,
                                       n_jobs=-1)
    boruta_selector = BorutaPy(estimator, n_estimators='auto',
                               random_state=config.get("random_state", 42))
    boruta_selector.fit(self.train_X.values, self.train_Y)
    selected_features = self.train_X.columns[boruta_selector.support_].tolist()
    
    self.SELECTED_FEATURE_LIST = selected_features
    apply_feature_selection(self)

# 4. RFECV Feature Selection
def rfecv_feature_selection(self):
    config = self.feature_selection_config
    self.feature_selection_config["name"] = "rfecv_feature_selection"

    n_estimators = config.get("n_estimators", 100)
    random_state = config.get("random_state", 42)
    estimator = RandomForestClassifier(n_estimators=n_estimators,
                                       random_state=random_state)
    rfecv_selector = RFECV(estimator,
                           step=1,
                           cv=config.get("cv", 5),
                           scoring=config.get("scoring", "accuracy"),
                           n_jobs=-1)
    rfecv_selector.fit(self.train_X, self.train_Y)

    selected_features = self.train_X.columns[rfecv_selector.support_].tolist()

    self.SELECTED_FEATURE_LIST = selected_features
    apply_feature_selection(self)

# 5. Variance Threshold Feature Selection
def variance_threshold_selection(self):
    config = self.feature_selection_config
    self.feature_selection_config["name"] = "variance_threshold_selection"

    threshold = config.get("threshold", 0.0)
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(self.train_X)
    selected_features = self.train_X.columns[selector.get_support()].tolist()

    self.SELECTED_FEATURE_LIST = selected_features
    apply_feature_selection(self)

# 7. L1-based Feature Selection
def l1_feature_selection(self):
    config = self.feature_selection_config
    self.feature_selection_config["name"] = "l1_feature_selection"

    C = config.get("C", 1.0)
    max_iter=config.get("max_iter", 1000)
    random_state=config.get("random_state", 42)

    estimator = LogisticRegression(penalty='l1',
                                   solver='saga',
                                   C=C,
                                   max_iter=max_iter,
                                   random_state=random_state)
    estimator.fit(self.train_X, self.train_Y)
    # 檢查 train_X 是否有足夠的特徵供選取
    if self.train_X.shape[1] < 2:
        print("資料特徵數不足以執行 SequentialFeatureSelector (至少需2個特徵)，跳過特徵選取")
        return 0
    from sklearn.feature_selection import SelectFromModel
    selector = SelectFromModel(estimator, prefit=True)
    selected_features = self.train_X.columns[selector.get_support()].tolist()

    self.SELECTED_FEATURE_LIST = selected_features
    apply_feature_selection(self)