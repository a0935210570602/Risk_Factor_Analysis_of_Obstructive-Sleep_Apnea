from second_recurrent_prediction import SecondStrokePrediction
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", message="The least populated class in y has only")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

file_path = 'raw_data/age_below_65.csv'
model_prediction = SecondStrokePrediction(file_path)

model_prediction.load_data()
model_prediction.standardize_data()
model_prediction.smote_borderline()
model_prediction.sfs_logistic_feature_selection()

model_prediction.predict_svm_linear()
model_prediction.predict_svm_poly()
model_prediction.predict_svm_rbf()
model_prediction.predict_decision_tree()
model_prediction.predict_random_forest()
model_prediction.predict_adaboost()
model_prediction.predict_gradient_boost()
model_prediction.predict_xgboost()

############################################


model_prediction.load_data()
model_prediction.standardize_data()
model_prediction.smote_borderline()
model_prediction.sfs_knn_feature_selection()
 
model_prediction.predict_svm_linear()
model_prediction.predict_svm_poly()
model_prediction.predict_svm_rbf()
model_prediction.predict_decision_tree()
model_prediction.predict_random_forest()
model_prediction.predict_adaboost()
model_prediction.predict_gradient_boost()
model_prediction.predict_xgboost()

############################################

model_prediction.load_data()
model_prediction.standardize_data()
model_prediction.smote_borderline()
model_prediction.sfs_linear_feature_selection()

model_prediction.predict_svm_linear()
model_prediction.predict_svm_poly()
model_prediction.predict_svm_rbf()
model_prediction.predict_decision_tree()
model_prediction.predict_random_forest()
model_prediction.predict_adaboost()
model_prediction.predict_gradient_boost()
model_prediction.predict_xgboost()

############################################

model_prediction.load_data()
model_prediction.standardize_data()
model_prediction.smote_borderline()
model_prediction.sfs_random_forest_feature_selection()

model_prediction.predict_svm_linear()
model_prediction.predict_svm_poly()
model_prediction.predict_svm_rbf()
model_prediction.predict_decision_tree()
model_prediction.predict_random_forest()
model_prediction.predict_adaboost()
model_prediction.predict_gradient_boost()
model_prediction.predict_xgboost()

############################################

model_prediction.load_data()
model_prediction.standardize_data()
model_prediction.smote_borderline()
model_prediction.sfs_xgboost_feature_selection()

model_prediction.predict_svm_linear()
model_prediction.predict_svm_poly()
model_prediction.predict_svm_rbf()
model_prediction.predict_decision_tree()
model_prediction.predict_random_forest()
model_prediction.predict_adaboost()
model_prediction.predict_gradient_boost()
model_prediction.predict_xgboost()

############################################

model_prediction.load_data()
model_prediction.standardize_data()
model_prediction.smote_borderline()
model_prediction.permutation_feature_selection()

model_prediction.predict_svm_linear()
model_prediction.predict_svm_poly()
model_prediction.predict_svm_rbf()
model_prediction.predict_decision_tree()
model_prediction.predict_random_forest()
model_prediction.predict_adaboost()
model_prediction.predict_gradient_boost()
model_prediction.predict_xgboost()

############################################

model_prediction.load_data()
model_prediction.standardize_data()
model_prediction.smote_borderline()
model_prediction.boruta_feature_selection()

model_prediction.predict_svm_linear()
model_prediction.predict_svm_poly()
model_prediction.predict_svm_rbf()
model_prediction.predict_decision_tree()
model_prediction.predict_random_forest()
model_prediction.predict_adaboost()
model_prediction.predict_gradient_boost()
model_prediction.predict_xgboost()

############################################

model_prediction.load_data()
model_prediction.standardize_data()
model_prediction.smote_borderline()
model_prediction.rfecv_feature_selection()

model_prediction.predict_svm_linear()
model_prediction.predict_svm_poly()
model_prediction.predict_svm_rbf()
model_prediction.predict_decision_tree()
model_prediction.predict_random_forest()
model_prediction.predict_adaboost()
model_prediction.predict_gradient_boost()
model_prediction.predict_xgboost()

############################################

model_prediction.load_data()
model_prediction.standardize_data()
model_prediction.smote_borderline()
model_prediction.variance_threshold_selection()

model_prediction.predict_svm_linear()
model_prediction.predict_svm_poly()
model_prediction.predict_svm_rbf()
model_prediction.predict_decision_tree()
model_prediction.predict_random_forest()
model_prediction.predict_adaboost()
model_prediction.predict_gradient_boost()
model_prediction.predict_xgboost()

############################################

model_prediction.load_data()
model_prediction.standardize_data()
model_prediction.smote_borderline()
model_prediction.l1_feature_selection()

model_prediction.predict_svm_linear()
model_prediction.predict_svm_poly()
model_prediction.predict_svm_rbf()
model_prediction.predict_decision_tree()
model_prediction.predict_random_forest()
model_prediction.predict_adaboost()
model_prediction.predict_gradient_boost()
model_prediction.predict_xgboost()

############################################