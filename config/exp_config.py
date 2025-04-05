# config.py

# 定義資料檔案路徑
age_below_65_path = 'raw_data/age_below_65.csv'
age_between_65_80_path = 'raw_data/age_between_65_80.csv'
age_over_80_path = 'raw_data/age_over_80.csv'

# 定義實驗配置，包含 data_config 與 model_config
exp_config = [
    {
        "data_config": {
            "path": age_below_65_path,
            "test_size": 0.2,
            "random_state": 42
        },
        "model_config": [
            {"model_name": "svm_linear", "params": {"C": 1.0, "max_iter": 3000}, "runs": 3},
            {"model_name": "svm_linear", "params": {"C": 0.5, "max_iter": 1500}, "runs": 3},
        ]
    }
]
