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

# 定義模型配置
model_configs = [
    {"model_name": "svm_linear", "params": {"C": 1.0, "max_iter": 3000}, "runs": 3},
    {"model_name": "svm_linear", "params": {"C": 1.0, "max_iter": 1000}, "runs": 1},
    # {"name": "xgb", "params": {"n_estimators": 100, "max_depth": 4}, "runs": 3},
    # {"name": "decision_tree", "params": {"max_depth": 3}, "runs": 3},
    # {"name": "decision_tree", "params": {"max_depth": 5}, "runs": 3},
    # {"name": "decision_tree", "params": {"max_depth": None}, "runs": 3},
]

import numpy as np
from sklearn.datasets import make_classification

# 建立合成數據
X_train, y_train = make_classification(n_samples=100, n_features=20, random_state=42)
X_test, y_test = make_classification(n_samples=30, n_features=20, random_state=43)

# 初始化並執行實驗流程控制器
pipeline = ExperimentPipeline(model_configs, X_train, y_train, X_test, y_test)
results = pipeline.run()

for res in results:
    print(res)