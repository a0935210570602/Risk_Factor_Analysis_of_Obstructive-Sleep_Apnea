import pandas as pd
import os

class SecondStrokePrediction:
    PATH = os.path.join(os. getcwd(), 'result_data')
    SELECTED_FEATURE_LIST = ['age', 'sex', 'HLOS', 'NIHSS', 'tPA(0/1)', 'EVT(0/1)',
        'HTN(0/1)', 'DM(0/1)', 'Dyslipidemia(0/1)', 'Af(0/1)', 'smoking(Y/N/Q)',
        'LDL ', 'cholesterol', 'TG', 'Cre', 'SGPT', 'HbA1c', 'MRS']
    LABEL_NAME = ['Second_Stroke']
    RESULT_DF_COLUMN = ['Model name','dataset','accuracy','precision','recall','f1-score','auc','feature','balance_config','feature_selection_config']
    TEN_FOLD_DF_COLUMN = ['Model name', 'dataset', 'accuracy_mean', 'accuracy_std',
        'precision_mean', 'precision_std', 'recall_mean', 'recall_std',
        'f1-score_mean', 'f1-score_std', 'auc_mean', 'auc_std','feature','balance_config','feature_selection_config']
    ROC_CURVE_RESULT_COLUMN = ['model', 'dataset', 'auc', 'fpr', 'tpr']
    RANDOM_SEED = 42
    data_config = { "path": None, "train_size": 0.8, "random_state": 42, "down_sampling_rate":1}
    balance_config = {"name": None, "sample amount": None, "random_state": 42, "neighbors": 5}
    feature_selection_config = {
        "name": None,
        "estimator": "logistic",     # 以 LogisticRegression 作為基底
        "direction": "forward", 
        "n_neighbors":5,    # 選擇後向淘汰法 (也可以設定 'forward')
        "n_features_to_select": "auto",   # 選取 5 個最佳特徵
        "max_iter": 1000, 
        "max_depth":5,
        "learning_rate":0.1,     # LogisticRegression 的最大迭代次數
        "random_state": 42           # 隨機種子，確保重現性
    }


    def __init__(self, file_path: str):
        self.data_config["path"] = os.path.join(os.getcwd(), file_path)
        # self.data_path = os.path.join(os.getcwd(), file_path)
        self.data_df = pd.DataFrame
        self.train_X = pd.DataFrame
        self.train_Y = pd.DataFrame
        self.valid_X = pd.DataFrame
        self.valid_Y = pd.DataFrame
        self.data_X = pd.DataFrame
        self.data_Y = pd.DataFrame
        self.train_result_df = pd.DataFrame(columns=self.RESULT_DF_COLUMN)
        self.test_result_df = pd.DataFrame(columns=self.RESULT_DF_COLUMN)
        self.roc_curve_result_df = pd.DataFrame(columns=self.ROC_CURVE_RESULT_COLUMN)
        self.ten_fold_avg_std_df = pd.DataFrame
        self.standardization_or_not = False

    from ._predict import predict_svm_linear
    from ._predict import predict_svm_poly
    from ._predict import predict_svm_rbf
    from ._predict import predict_decision_tree
    from ._predict import predict_random_forest
    from ._predict import predict_xgboost
    from ._predict import predict_adaboost
    from ._predict import predict_gradient_boost
    from ._predict import predict_dcnn
    from ._predict import show_all_result
    from ._predict import set_prediction_model
    from ._data_processing import load_data
    from ._data_processing import apply_standardization
    from ._data_processing import set_downsampling_rate
    from ._data_processing import set_smote_method
    # from ._data_processing import standardize_data
    from ._data_processing import cross_validation
    from ._data_processing import prepare_tenfold_data
    from ._data_processing import display_total_result_train
    from ._data_processing import display_total_result_test
    from ._data_processing import display_total_ten_fold_result
    from ._data_processing import clear_data_and_model
    from ._balance_data import smote_smotenc
    from ._balance_data import smote_standard
    from ._balance_data import smote_svm
    from ._balance_data import smote_borderline
    from ._balance_data import smote_adasyn
    from ._feature_select import sfs_logistic_feature_selection
    from ._feature_select import sfs_knn_feature_selection
    from ._feature_select import sfs_linear_feature_selection
    from ._feature_select import sfs_random_forest_feature_selection
    from ._feature_select import sfs_xgboost_feature_selection
    from ._feature_select import permutation_feature_selection
    from ._feature_select import boruta_feature_selection
    from ._feature_select import rfecv_feature_selection
    from ._feature_select import variance_threshold_selection
    from ._feature_select import l1_feature_selection
    from ._feature_analyze import pca
    from ._feature_analyze import tsne
    from ._model import dcnn_fit
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
    from ._result import gen_result