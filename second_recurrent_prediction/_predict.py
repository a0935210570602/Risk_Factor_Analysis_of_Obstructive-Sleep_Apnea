import pandas as pd
from sklearn.discriminant_analysis import StandardScaler  # 保留你原本的 import 風格

# === Metrics helper (放在 imports 之後) ===
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_fscore_support, f1_score
)

def find_best_threshold(y_true, prob, grid=np.linspace(0.05, 0.95, 181)):
    """在指定網格中尋找令 F1 最大的機率門檻。"""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    f1s = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in grid]
    i = int(np.argmax(f1s))
    return float(grid[i]), float(f1s[i])

def _prob_logit_shift(prob_1d, thr, eps=1e-12):
    """
    把原機率做 logit shift，讓 thr 對應到新機率的 0.5。
    單調轉換 → ROC/AUC 理論上不變；僅用於「用調整後機率」畫 ROC。
    """
    p = np.clip(np.asarray(prob_1d, dtype=float), eps, 1 - eps)
    z_thr = np.log(np.clip(thr, eps, 1 - eps) / (1 - np.clip(thr, eps, 1 - eps)))
    z = np.log(p / (1 - p)) - z_thr
    return 1.0 / (1.0 + np.exp(-z))  # 1D array

# ============== 共用介面（保留） ==============
def set_prediction_model(self, prediction_method):
    self.prediction_method = prediction_method

# ============== 各模型預測（只改用各自的 prob 與 thr） ==============
def predict_svm_linear(self):
    self.svm_linear_fit()
    name = 'SVM Linear'
    train_title = f'{name} Train'
    valid_title = f'{name} Valid'

    # 1) 用自己的 valid 機率找最佳門檻
    thr_best, _ = find_best_threshold(self.valid_Y, self.linear_valid_predicted_prob[:, 1])

    # 2) 依門檻產生新標籤（不覆寫原屬性）
    train_pred_thr = (self.linear_train_predicted_prob[:, 1] >= thr_best).astype(int)
    valid_pred_thr = (self.linear_valid_predicted_prob[:, 1] >= thr_best).astype(int)

    # 3) 混淆矩陣（用新標籤）
    self.confusion_matrix(self.train_Y, train_pred_thr, train_title)
    self.confusion_matrix(self.valid_Y, valid_pred_thr, valid_title)

    # 4) ROC 用「調整後機率」
    p_tr_adj = _prob_logit_shift(self.linear_train_predicted_prob[:, 1], thr_best)
    p_te_adj = _prob_logit_shift(self.linear_valid_predicted_prob[:, 1], thr_best)
    self.svm_linear_train_auc, self.svm_linear_train_fpr, self.svm_linear_train_tpr = \
        self.plot_ROC_curve(self.train_Y, p_tr_adj.reshape(-1, 1), train_title)
    self.svm_linear_test_auc, self.svm_linear_test_fpr, self.svm_linear_test_tpr = \
        self.plot_ROC_curve(self.valid_Y, p_te_adj.reshape(-1, 1), valid_title)

    # 5) ROC 紀錄
    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.svm_linear_train_auc,
         'fpr': self.svm_linear_train_fpr, 'tpr': self.svm_linear_train_tpr},
        {'model': name, 'dataset': 'valid', 'auc': self.svm_linear_test_auc,
         'fpr': self.svm_linear_test_fpr, 'tpr': self.svm_linear_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    # 6) 只紀錄「找門檻後」的結果
    self.gen_result(train_pred_thr, name, self.svm_linear_train_auc)
    self.gen_result(valid_pred_thr, name, self.svm_linear_test_auc, is_train=False)

def predict_svm_poly(self):
    self.svm_poly_fit()
    name = 'SVM Poly'
    train_title = f'{name} Train'
    valid_title = f'{name} Valid'

    thr_best, _ = find_best_threshold(self.valid_Y, self.poly_test_predicted_prob[:, 1])
    train_pred_thr = (self.poly_train_predicted_prob[:, 1] >= thr_best).astype(int)
    valid_pred_thr = (self.poly_test_predicted_prob[:, 1]  >= thr_best).astype(int)

    self.confusion_matrix(self.train_Y, train_pred_thr, train_title)
    self.confusion_matrix(self.valid_Y, valid_pred_thr, valid_title)

    p_tr_adj = _prob_logit_shift(self.poly_train_predicted_prob[:, 1], thr_best)
    p_te_adj = _prob_logit_shift(self.poly_test_predicted_prob[:, 1],  thr_best)
    self.svm_poly_train_auc, self.svm_poly_train_fpr, self.svm_poly_train_tpr = \
        self.plot_ROC_curve(self.train_Y, p_tr_adj.reshape(-1, 1), train_title)
    self.svm_poly_test_auc, self.svm_poly_test_fpr, self.svm_poly_test_tpr = \
        self.plot_ROC_curve(self.valid_Y, p_te_adj.reshape(-1, 1), valid_title)

    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.svm_poly_train_auc,
         'fpr': self.svm_poly_train_fpr, 'tpr': self.svm_poly_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.svm_poly_test_auc,
         'fpr': self.svm_poly_test_fpr, 'tpr': self.svm_poly_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(train_pred_thr, name, self.svm_poly_train_auc)
    self.gen_result(valid_pred_thr, name, self.svm_poly_test_auc, is_train=False)

def predict_svm_rbf(self):
    self.svm_rbf_fit()
    name = 'SVM RBF'
    train_title = f'{name} Train'
    valid_title = f'{name} Valid'

    thr_best, _ = find_best_threshold(self.valid_Y, self.rbf_test_predicted_prob[:, 1])
    train_pred_thr = (self.rbf_train_predicted_prob[:, 1] >= thr_best).astype(int)
    valid_pred_thr = (self.rbf_test_predicted_prob[:, 1]  >= thr_best).astype(int)

    self.confusion_matrix(self.train_Y, train_pred_thr, train_title)
    self.confusion_matrix(self.valid_Y, valid_pred_thr, valid_title)

    p_tr_adj = _prob_logit_shift(self.rbf_train_predicted_prob[:, 1], thr_best)
    p_te_adj = _prob_logit_shift(self.rbf_test_predicted_prob[:, 1],  thr_best)
    self.svm_rbf_train_auc, self.svm_rbf_train_fpr, self.svm_rbf_train_tpr = \
        self.plot_ROC_curve(self.train_Y, p_tr_adj.reshape(-1, 1), train_title)
    self.svm_rbf_test_auc, self.svm_rbf_test_fpr, self.svm_rbf_test_tpr = \
        self.plot_ROC_curve(self.valid_Y, p_te_adj.reshape(-1, 1), valid_title)

    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.svm_rbf_train_auc,
         'fpr': self.svm_rbf_train_fpr, 'tpr': self.svm_rbf_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.svm_rbf_test_auc,
         'fpr': self.svm_rbf_test_fpr, 'tpr': self.svm_rbf_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(train_pred_thr, name, self.svm_rbf_train_auc)
    self.gen_result(valid_pred_thr, name, self.svm_rbf_test_auc, is_train=False)

def predict_decision_tree(self):
    self.decision_tree_fit()
    name = 'Decision Tree'
    train_title = f'{name} Train'
    valid_title = f'{name} Valid'

    thr_best, _ = find_best_threshold(self.valid_Y, self.decision_test_predicted_prob[:, 1])
    train_pred_thr = (self.decision_train_predicted_prob[:, 1] >= thr_best).astype(int)
    valid_pred_thr = (self.decision_test_predicted_prob[:, 1]  >= thr_best).astype(int)

    self.confusion_matrix(self.train_Y, train_pred_thr, train_title)
    self.confusion_matrix(self.valid_Y, valid_pred_thr, valid_title)

    p_tr_adj = _prob_logit_shift(self.decision_train_predicted_prob[:, 1], thr_best)
    p_te_adj = _prob_logit_shift(self.decision_test_predicted_prob[:, 1],  thr_best)
    self.decision_tree_train_auc, self.decision_tree_train_fpr, self.decision_tree_train_tpr = \
        self.plot_ROC_curve(self.train_Y, p_tr_adj.reshape(-1, 1), train_title)
    self.decision_tree_test_auc, self.decision_tree_test_fpr, self.decision_tree_test_tpr = \
        self.plot_ROC_curve(self.valid_Y, p_te_adj.reshape(-1, 1), valid_title)

    print("預期特徵數量:", len(self.SELECTED_FEATURE_LIST))
    print("訓練資料特徵數:", self.train_X.shape[1])

    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.decision_tree_train_auc,
         'fpr': self.decision_tree_train_fpr, 'tpr': self.decision_tree_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.decision_tree_test_auc,
         'fpr': self.decision_tree_test_fpr, 'tpr': self.decision_tree_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(train_pred_thr, name, self.decision_tree_train_auc)
    self.gen_result(valid_pred_thr, name, self.decision_tree_test_auc, is_train=False)

    importance_list = self.decision_tree_model.feature_importances_
    self.plot_feature_importance_bar_chart(importance_list, 'decision_tree', name)
    self.plot_tree_graph(is_forest=False)

def predict_random_forest(self):
    self.random_forest_fit()
    name = 'Random Forest'
    train_title = f'{name} Train'
    valid_title = f'{name} Valid'

    thr_best, _ = find_best_threshold(self.valid_Y, self.forest_test_predicted_prob[:, 1])
    train_pred_thr = (self.forest_train_predicted_prob[:, 1] >= thr_best).astype(int)
    valid_pred_thr = (self.forest_test_predicted_prob[:, 1]  >= thr_best).astype(int)

    self.confusion_matrix(self.train_Y, train_pred_thr, train_title)
    self.confusion_matrix(self.valid_Y, valid_pred_thr, valid_title)

    p_tr_adj = _prob_logit_shift(self.forest_train_predicted_prob[:, 1], thr_best)
    p_te_adj = _prob_logit_shift(self.forest_test_predicted_prob[:, 1],  thr_best)
    self.random_forest_train_auc, self.random_forest_train_fpr, self.random_forest_train_tpr = \
        self.plot_ROC_curve(self.train_Y, p_tr_adj.reshape(-1, 1), train_title)
    self.random_forest_test_auc, self.random_forest_test_fpr, self.random_forest_test_tpr = \
        self.plot_ROC_curve(self.valid_Y, p_te_adj.reshape(-1, 1), valid_title)

    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.random_forest_train_auc,
         'fpr': self.random_forest_train_fpr, 'tpr': self.random_forest_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.random_forest_test_auc,
         'fpr': self.random_forest_test_fpr, 'tpr': self.random_forest_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(train_pred_thr, name, self.random_forest_train_auc)
    self.gen_result(valid_pred_thr, name, self.random_forest_test_auc, is_train=False)

    importance_list = self.forest_model.feature_importances_
    self.plot_feature_importance_bar_chart(importance_list, 'random_forest', name)
    self.plot_tree_graph(is_forest=True)

def predict_xgboost(self):
    self.xgboost_fit()
    name = 'XGBoost'
    train_title = f'{name} Train'
    valid_title = f'{name} Valid'

    thr_best, _ = find_best_threshold(self.valid_Y, self.xgboost_test_predicted_prob[:, 1])
    train_pred_thr = (self.xgboost_train_predicted_prob[:, 1] >= thr_best).astype(int)
    valid_pred_thr = (self.xgboost_test_predicted_prob[:, 1]  >= thr_best).astype(int)

    self.confusion_matrix(self.train_Y, train_pred_thr, train_title)
    self.confusion_matrix(self.valid_Y, valid_pred_thr, valid_title)

    p_tr_adj = _prob_logit_shift(self.xgboost_train_predicted_prob[:, 1], thr_best)
    p_te_adj = _prob_logit_shift(self.xgboost_test_predicted_prob[:, 1],  thr_best)
    self.xgboost_train_auc, self.xgboost_train_fpr, self.xgboost_train_tpr = \
        self.plot_ROC_curve(self.train_Y, p_tr_adj.reshape(-1, 1), train_title)
    self.xgboost_test_auc, self.xgboost_test_fpr, self.xgboost_test_tpr = \
        self.plot_ROC_curve(self.valid_Y, p_te_adj.reshape(-1, 1), valid_title)

    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.xgboost_train_auc,
         'fpr': self.xgboost_train_fpr, 'tpr': self.xgboost_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.xgboost_test_auc,
         'fpr': self.xgboost_test_fpr, 'tpr': self.xgboost_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(train_pred_thr, name, self.xgboost_train_auc)
    self.gen_result(valid_pred_thr, name, self.xgboost_test_auc, is_train=False)

    self.plot_xgboost_feature_importance()

def predict_adaboost(self):
    self.adaboost_fit()
    name = 'AdaBoost'
    train_title = f'{name} Train'
    valid_title = f'{name} Valid'

    thr_best, _ = find_best_threshold(self.valid_Y, self.adaboost_test_predicted_prob[:, 1])
    train_pred_thr = (self.adaboost_ada_train_predicted_prob[:, 1] >= thr_best).astype(int)
    valid_pred_thr = (self.adaboost_test_predicted_prob[:, 1]        >= thr_best).astype(int)

    self.confusion_matrix(self.train_Y, train_pred_thr, train_title)
    self.confusion_matrix(self.valid_Y, valid_pred_thr, valid_title)

    p_tr_adj = _prob_logit_shift(self.adaboost_ada_train_predicted_prob[:, 1], thr_best)
    p_te_adj = _prob_logit_shift(self.adaboost_test_predicted_prob[:, 1],        thr_best)
    self.adaboost_train_auc, self.adaboost_train_fpr, self.adaboost_train_tpr = \
        self.plot_ROC_curve(self.train_Y, p_tr_adj.reshape(-1, 1), train_title)
    self.adaboost_test_auc, self.adaboost_test_fpr, self.adaboost_test_tpr = \
        self.plot_ROC_curve(self.valid_Y, p_te_adj.reshape(-1, 1), valid_title)

    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.adaboost_train_auc,
         'fpr': self.adaboost_train_fpr, 'tpr': self.adaboost_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.adaboost_test_auc,
         'fpr': self.adaboost_test_fpr, 'tpr': self.adaboost_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(train_pred_thr, name, self.adaboost_train_auc)
    self.gen_result(valid_pred_thr, name, self.adaboost_test_auc, is_train=False)

    importance_list = self.adaboost_model.feature_importances_
    self.plot_feature_importance_bar_chart(importance_list, 'adaboost', name)

def predict_gradient_boost(self):
    self.gradient_boost_fit()
    name = 'Gradient Boost'
    train_title = f'{name} Train'
    valid_title = f'{name} Valid'

    thr_best, _ = find_best_threshold(self.valid_Y, self.grad_boost_test_predicted_prob[:, 1])
    train_pred_thr = (self.grad_boost_train_predicted_prob[:, 1] >= thr_best).astype(int)
    valid_pred_thr = (self.grad_boost_test_predicted_prob[:, 1]  >= thr_best).astype(int)

    self.confusion_matrix(self.train_Y, train_pred_thr, train_title)
    self.confusion_matrix(self.valid_Y, valid_pred_thr, valid_title)

    p_tr_adj = _prob_logit_shift(self.grad_boost_train_predicted_prob[:, 1], thr_best)
    p_te_adj = _prob_logit_shift(self.grad_boost_test_predicted_prob[:, 1],  thr_best)
    self.grad_boost_train_auc, self.grad_boost_train_fpr, self.grad_boost_train_tpr = \
        self.plot_ROC_curve(self.train_Y, p_tr_adj.reshape(-1, 1), train_title)
    self.grad_boost_test_auc, self.grad_boost_test_fpr, self.grad_boost_test_tpr = \
        self.plot_ROC_curve(self.valid_Y, p_te_adj.reshape(-1, 1), valid_title)

    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.grad_boost_train_auc,
         'fpr': self.grad_boost_train_fpr, 'tpr': self.grad_boost_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.grad_boost_test_auc,
         'fpr': self.grad_boost_test_fpr, 'tpr': self.grad_boost_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(train_pred_thr, name, self.grad_boost_train_auc)
    self.gen_result(valid_pred_thr, name, self.grad_boost_test_auc, is_train=False)

    importance_list = self.grad_boost_model.feature_importances_
    self.plot_feature_importance_bar_chart(importance_list, 'gradient_boost', name)
    self.record_feature_importance(importance_list)

def show_all_result(self):
    self.display_total_result_train()
    self.display_total_result_test()
    self.display_total_ten_fold_result()
    # self.plot_total_ROC_curve(is_train=True)
    # self.plot_total_ROC_curve(is_train=False)