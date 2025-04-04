import pandas as pd
import os

class SecondStrokePrediction:
    PATH = os.path.join(os. getcwd(), 'result_data')
    SELECTED_FEATURE_LIST = ['age', 'sex', 'HLOS', 'NIHSS', 'tPA(0/1)', 'EVT(0/1)',
        'HTN(0/1)', 'DM(0/1)', 'Dyslipidemia(0/1)', 'Af(0/1)', 'smoking(Y/N/Q)',
        'LDL ', 'cholesterol', 'TG', 'Cre', 'SGPT', 'HbA1c', 'MRS']
    LABEL_NAME = ['Second_Stroke']
    RESULT_DF_COLUMN = ['Model name','dataset','accuracy','precision','recall','f1-score','auc']
    TEN_FOLD_DF_COLUMN = ['Model name', 'dataset', 'accuracy_mean', 'accuracy_std',
        'precision_mean', 'precision_std', 'recall_mean', 'recall_std',
        'f1-score_mean', 'f1-score_std', 'auc_mean', 'auc_std']
    ROC_CURVE_RESULT_COLUMN = ['model', 'dataset', 'auc', 'fpr', 'tpr']

    def __init__(self, normal_filename: str, stroke_filename: str):
        self.normal_path = os.path.join(os.getcwd(), normal_filename)
        self.stroke_path = os.path.join(os.getcwd(), stroke_filename)
        self.normal_df = pd.read_csv(self.normal_path)
        self.stroke_df = pd.read_csv(self.stroke_path)
        self.data_df = pd.DataFrame
        self.train_X = pd.DataFrame
        self.train_Y = pd.DataFrame
        self.test_X = pd.DataFrame
        self.test_Y = pd.DataFrame
        self.data_X = pd.DataFrame
        self.data_Y = pd.DataFrame
        self.train_result_df = pd.DataFrame(columns=self.RESULT_DF_COLUMN)
        self.test_result_df = pd.DataFrame(columns=self.RESULT_DF_COLUMN)
        self.roc_curve_result_df = pd.DataFrame(columns=self.ROC_CURVE_RESULT_COLUMN)
        self.ten_fold_avg_std_df = pd.DataFrame

    from ._predict import predict_svm_linear
    from ._predict import predict_svm_poly
    from ._predict import predict_svm_rbf
    from ._predict import predict_decision_tree
    from ._predict import predict_random_forest
    from ._predict import predict_xgboost
    from ._predict import predict_adaboost
    from ._predict import predict_gradient_boost
    from ._predict import show_all_result
    from ._data_processing import load_data
    from ._data_processing import cross_validation
    from ._data_processing import gen_result
    from ._data_processing import display_total_result_train
    from ._data_processing import display_total_result_test
    from ._data_processing import display_total_ten_fold_result
    from ._feature_analyze import pca
    from ._feature_analyze import tsne
    from ._model import k_means_fit
    from ._model import svm_linear_fit
    from ._model import svm_poly_fit
    from ._model import svm_rbf_fit
    from ._model import decision_tree_fit
    from ._model import random_forest_fit
    from ._model import xgboost_fit
    from ._model import adaboost_fit
    from ._model import gradient_boost_fit
    from ._plot import confusion_matrix
    from ._plot import plot_ROC_curve
    from ._plot import plot_feature_importance_bar_chart
    from ._plot import plot_tree_graph
    from ._plot import plot_xgboost_feature_importance
    from ._plot import plot_total_ROC_curve