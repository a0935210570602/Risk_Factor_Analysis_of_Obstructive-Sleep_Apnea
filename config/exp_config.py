# config.py

# 定義資料檔案路徑
age_below_65_path = 'raw_data/age_below_65.csv'
age_between_65_80_path = 'raw_data/age_between_65_80.csv'
age_over_80_path = 'raw_data/age_over_80.csv'
old_data = 'raw_data/old_data.csv'

# 定義實驗配置，包含 data_config 與 model_config
MODEL_CONFIG = [
            {"model_name": "svm_linear", "params": {"C": 1.0, "kernel": "linear", "max_iter": 3000}, "runs": 1},
            {"model_name": "svm_linear", "params": {"C": 1.0, "kernel": "poly", "max_iter": 3000}, "runs": 1},
            {"model_name": "svm_linear", "params": {"C": 1.0, "kernel": "rbf", "max_iter": 3000}, "runs": 1},
            {"model_name": "svm_linear", "params": {"C": 1.0, "kernel": "sigmoid", "max_iter": 3000}, "runs": 1},
            {"model_name": "decision_tree", "params": {"criterion": "gini", "max_depth": 3, "splitter": "best"}, "runs": 2},
            {"model_name": "decision_tree", "params": {"criterion": "gini", "max_depth":5, "splitter": "best"}, "runs": 2},
            {"model_name": "decision_tree", "params": {"criterion": "gini", "max_depth":7, "splitter": "best"}, "runs": 2},
            {"model_name": "decision_tree", "params": {"criterion": "gini", "max_depth": 10, "splitter": "best"}, "runs": 2},
            {"model_name": "decision_tree", "params": {"criterion": "entropy", "max_depth": 3, "splitter": "random"}, "runs": 2},
            {"model_name": "decision_tree", "params": {"criterion": "entropy", "max_depth":5, "splitter": "random"}, "runs": 2},
            {"model_name": "decision_tree", "params": {"criterion": "entropy", "max_depth":7, "splitter": "random"}, "runs": 2},
            {"model_name": "decision_tree", "params": {"criterion": "entropy", "max_depth": 10, "splitter": "random"}, "runs": 2},
            {"model_name": "random_forest", "params": {"n_estimators": 300, "criterion": "entropy", "max_depth": 7, "random_state": 42}, "runs": 2},
            {"model_name": "random_forest", "params": {"n_estimators": 400, "criterion": "entropy", "max_depth": 7, "random_state": 42}, "runs": 2},
            {"model_name": "random_forest", "params": {"n_estimators": 500, "criterion": "entropy", "max_depth": 7, "random_state": 42}, "runs": 2},
            {"model_name": "xgboost", "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}, "runs": 1},
            {"model_name": "adaboost", "params": {"n_estimators": 100, "learning_rate": 1.0, "algorithm": "SAMME.R", "random_state": 42}, "runs": 1},
            {"model_name": "gradient_boost", "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}, "runs": 1},

        ]

EXP_CONFIG = [
    {
        "data_config": {
            "path": age_below_65_path,
            "test_size": 0.2,
            "random_state": 40,
            "balance": "smote"  # 可選值: None 或 "smote"
        },
        "feature_select_config": {
            "method": "knn",
            "n_features_to_select": 'auto',
            "direction": "backward",
            "n_neighbors": 2
        },
        "model_config": MODEL_CONFIG
    },
    # {
    #     "data_config": {
    #         "path": age_below_65_path,
    #         "test_size": 0.2,
    #         "random_state": 41,
    #         "balance": "smote"  # 可選值: None 或 "smote"
    #     },
    #     "feature_select_config": {
    #         "method": "knn",
    #         "n_features_to_select": 'auto',
    #         "direction": "backward",
    #         "n_neighbors": 2
    #     },
    #     "model_config": MODEL_CONFIG
    # },
    # {
    #     "data_config": {
    #         "path": age_below_65_path,
    #         "test_size": 0.2,
    #         "random_state": 39,
    #         "balance": "smote"  # 可選值: None 或 "smote"
    #     },
    #     "feature_select_config": {
    #         "method": "knn",
    #         "n_features_to_select": 'auto',
    #         "direction": "backward",
    #         "n_neighbors": 2
    #     },
    #     "model_config": MODEL_CONFIG
    # },
]
