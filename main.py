import numpy as np
import pandas as pd
from experiment_pipeline import ExperimentPipeline
from second_recurrent_prediction import SecondStrokePrediction

# normal_filename = 'raw_data/age_below_65_health.csv'
# stroke_filename = 'raw_data/age_below_65_stroke.csv'
# model_prediction = SecondStrokePrediction(normal_filename, stroke_filename)
# model_prediction.predict_svm_linear()
# model_prediction.predict_svm_poly()
# model_prediction.predict_svm_rbf()
# model_prediction.predict_decision_tree()
# model_prediction.predict_random_forest()
# model_prediction.predict_adaboost()
# model_prediction.predict_gradient_boost()
# model_prediction.predict_xgboost()
# model_prediction.show_all_result()
age_below_65_path = 'raw_data/age_below_65.csv'
age_between_65_80_path = 'raw_data/age_between_65_80.csv'
age_over_80_path = 'raw_data/age_over_80.csv'

# 定義模型配置
exp_config = [
    {
        "data_config": {
            "path": age_below_65_path,
            "test_size": 0.2,
            "random_state": 42
        },
        "model_config": 
            [
                {"model_name": "svm_linear", "params": {"C": 1.0, "max_iter": 3000}, "runs": 3},
                {"model_name": "svm_linear", "params": {"C": 1.0, "max_iter": 2000}, "runs": 3}
            ]
    },
    {
        "data_config": {
            "path": age_between_65_80_path,
            "test_size": 0.2,
            "random_state": 42
        },
        "model_config": 
            [
                {"model_name": "svm_linear", "params": {"C": 1.0, "max_iter": 1500}, "runs": 3},
                {"model_name": "svm_linear", "params": {"C": 0.5, "max_iter": 2000}, "runs": 3}
            ]
    }
]

# 初始化並執行實驗流程控制器
pipeline = ExperimentPipeline(exp_config)
results = pipeline.run()

# for res in results:
#     print(res)