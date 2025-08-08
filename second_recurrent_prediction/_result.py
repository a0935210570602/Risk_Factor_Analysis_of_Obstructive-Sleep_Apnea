import os
import numpy as np
import pandas as pd
from sklearn import metrics

def gen_result(self, pred_y, name, train_auc, is_train=True):
    true_y = self.train_Y if is_train else self.valid_Y
    label = 'train' if is_train else 'test'
    accuracy = round(metrics.accuracy_score(true_y, pred_y), 2)
    precision = round(metrics.precision_score(true_y, pred_y), 2)
    recall = round(metrics.recall_score(true_y, pred_y), 2)
    f1_score_value = round(metrics.f1_score(true_y, pred_y), 2)
    auc_curve = round(train_auc, 2)

    # 👉 計算 Specificity（TN / (TN + FP)）
    cm = metrics.confusion_matrix(true_y, pred_y)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = round(tn / (tn + fp), 2) if (tn + fp) > 0 else 0
    else:
        tn = fp = fn = tp = specificity = 0

    confusion_str = f"[{tp}, {fp}]\n[{fn}, {tn}]"
    filename = os.path.basename(self.data_config["path"])
    name_without_ext, _ = os.path.splitext(filename)
    dir_path = os.path.join(self.PATH, label)
    file_path = os.path.join(dir_path, f'{label}_{name_without_ext}.csv')

    # 確保資料夾存在
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 檢查目前已經有多少 fold 結果（不含這一筆）
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        curr_lines = len(existing_df)
    else:
        existing_df = None
        curr_lines = 0

    # 動態決定這一行的 group id
    group_id = (curr_lines // 10) + 1

    # 建立這一筆 fold 結果
    df = pd.DataFrame({
        'Model name': [name],
        'Fold': [self.fold],
        'Group': [group_id],
        'dataset': [label],
        'accuracy': [accuracy],
        'specificity': [specificity],
        'precision': [precision],
        'confusion_matrix': [confusion_str],
        'recall': [recall],
        'f1-score': [f1_score_value],
        'auc': [auc_curve],
        'feature': [self.SELECTED_FEATURE_LIST],
        'balance_config': [self.balance_config],
        'downsampling_rate': [self.data_config["down_sampling_rate"]],
        'feature_selection_config': [self.feature_selection_config],
    })

    # append or create
    if existing_df is not None:
        result_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        result_df = df.copy()

    if is_train:
        self.train_result_df = result_df
    else:
        self.test_result_df = result_df

    # 儲存 CSV
    result_df.to_csv(file_path, index=False)



    # summary 也跟著用正確的 group_id
    if self.fold == 10:
        self.plot_feature_summary('gradient_boost_feature_importance_summary')
        self.plot_feature_summary_top_five('gradient_boost_feature_importance_summary_top_five')
        # self.clean_feature_importance()
        all_results = pd.read_csv(file_path)
        recent_results = all_results[all_results["Group"] == group_id]  # 取本組
        metric_cols = ['accuracy', 'specificity', 'precision', 'recall', 'f1-score', 'auc']
        avg_metrics = recent_results[metric_cols].mean().round(3)
        last_row = recent_results.iloc[-1]
        summary = {
            'Model name': last_row['Model name'],
            'Group': group_id,
            'dataset': last_row['dataset'],
            'accuracy': avg_metrics['accuracy'],
            'specificity': avg_metrics['specificity'],
            'precision': avg_metrics['precision'],
            'confusion_matrix': 'avg',
            'recall': avg_metrics['recall'],
            'f1-score': avg_metrics['f1-score'],
            'auc': avg_metrics['auc'],
            'feature': last_row['feature'],
            'balance_config': last_row['balance_config'],
            'downsampling_rate': last_row['downsampling_rate'],
            'feature_selection_config': last_row['feature_selection_config'],
        }
        summary_file_path = os.path.join(dir_path, f'summary_{name_without_ext}.csv')
        pd.DataFrame([summary]).to_csv(
            summary_file_path,
            mode='a',
            header=not os.path.exists(summary_file_path),
            index=False
        )

def record_feature_importance(self, importances):
    feature_names = self.train_X.columns
    for f, imp in zip(feature_names, importances):
        self.feature_importance_dict[f].append(imp)

def clean_feature_importance(self):
    self.feature_importance_dict = {f: [] for f in self.SELECTED_FEATURE_LIST}