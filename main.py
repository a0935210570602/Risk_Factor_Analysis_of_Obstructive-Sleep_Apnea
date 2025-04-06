from pipeline.experiment_pipeline import ExperimentPipeline
from config.exp_config import EXP_CONFIG
import warnings
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning

warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", message="The least populated class in y has only")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

# 初始化並執行實驗流程控制器kj
pipeline = ExperimentPipeline(EXP_CONFIG)
results = pipeline.run()