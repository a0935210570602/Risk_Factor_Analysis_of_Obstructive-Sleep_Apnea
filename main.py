from second_recurrent_prediction import SecondStrokePrediction
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning
from exp_config import smote_methods, feature_selection_methods, prediction_methods

import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The least populated class in y has only")


file_path = 'raw_data/age_below_65_91_samples.csv'
model_prediction = SecondStrokePrediction(file_path)

# 對於每一種 SMOTE 與特徵選擇的組合
for data_smote in smote_methods:
    for data_feature_selection in feature_selection_methods:
        for model_predict in prediction_methods:
            # 每次實驗從原始資料開始：重新載入與標準化
            file_path = 'raw_data/age_below_65_91_samples.csv'
            model_prediction = SecondStrokePrediction(file_path)
            model_prediction.load_data()
            model_prediction.standardize_data()
            data_smote(model_prediction)
            data_feature_selection(model_prediction)
            model_predict(model_prediction)