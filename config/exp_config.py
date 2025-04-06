# config.py

# 定義資料檔案路徑
age_below_65_path = 'raw_data/age_below_65.csv'
age_between_65_80_path = 'raw_data/age_between_65_80.csv'
age_over_80_path = 'raw_data/age_over_80.csv'

# 定義實驗配置，包含 data_config 與 model_config
EXP_CONFIG = [
    {
        "data_config": {
            "path": age_below_65_path,
            "test_size": 0.2,
            "random_state": 42,
            "balance": "smote"  # 可選值: None 或 "smote"
        },
        "feature_select_config": {
            "method": "knn",
            "n_features_to_select": 'auto',
            "direction": "backward",
            "n_neighbors": 5
        },
        "model_config": [
            {"model_name": "svm_linear", "params": {"C": 1.0, "max_iter": 3000}, "runs": 1},
        ]
    },{
        "data_config": {
            "path": age_below_65_path,
            "test_size": 0.2,
            "random_state": 42,
            "balance": "smote"  # 可選值: None 或 "smote"
        },
        "feature_select_config": {
            "method": "knn",
            "n_features_to_select": 'auto',
            "direction": "forward",
            "n_neighbors": 5
        },
        "model_config": [
            {"model_name": "svm_linear", "params": {"C": 1.0, "max_iter": 3000}, "runs": 1},
        ]
    },
    {
        "data_config": {
            "path": age_below_65_path,
            "test_size": 0.2,
            "random_state": 42,
            "balance": "smote"  # 可選值: None 或 "smote"
        },
        "feature_select_config": {
            "method": "logistic",
            "n_features_to_select": "auto",
            "direction": "backward",
            "max_iter": 1500
        },
        "model_config": [
            {"model_name": "svm_linear", "params": {"C": 1.0, "max_iter": 3000}, "runs": 1},
        ]
    },{
        "data_config": {
            "path": age_below_65_path,
            "test_size": 0.2,
            "random_state": 42,
            "balance": "smote"  # 可選值: None 或 "smote"
        },
        "feature_select_config": {
            "method": "linear",
            "n_features_to_select": 5,
            "direction": "forward"
        },
        "model_config": [
            {"model_name": "svm_linear", "params": {"C": 1.0, "max_iter": 3000}, "runs": 1},
        ]
    },
]
