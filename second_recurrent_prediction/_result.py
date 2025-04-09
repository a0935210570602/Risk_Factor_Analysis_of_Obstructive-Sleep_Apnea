import os
import pandas as pd
from sklearn import metrics

def gen_result(self, pred_y, name, train_auc, is_train=True):
    true_y = self.train_Y if is_train else self.test_Y
    label = 'train' if is_train else 'test'
    accuracy = round(metrics.accuracy_score(true_y, pred_y), 2)
    precision = round(metrics.precision_score(true_y, pred_y), 2)
    recall = round(metrics.recall_score(true_y, pred_y), 2)
    f1_score_value = round(metrics.f1_score(true_y, pred_y), 2)
    auc_curve = round(train_auc, 2)

    df = pd.DataFrame({
        'Model name': [name],
        'dataset': [label],
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1-score': [f1_score_value],
        'auc': [auc_curve],
        'feature': [self.SELECTED_FEATURE_LIST],
        'balance_config': [self.balance_config],
        'downsampling_rate': [self.data_config["down_sampling_rate"]],
        'feature_selection_config': [self.feature_selection_config],
    })

    # 取得檔名：age_below_65.csv
    filename = os.path.basename(self.data_config["path"])
    # 分離副檔名，取得主檔名：age_below_65
    name_without_ext, _ = os.path.splitext(filename)
    # 決定目標資料夾與檔案路徑
    dir_path = os.path.join(self.PATH, label)
    file_path = os.path.join(dir_path, f'{label}_{name_without_ext}.csv')
    print(f'儲存結果到 {file_path}')
    # 確保資料夾存在
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 如果檔案已存在則讀取舊檔接續寫入
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        result_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        result_df = df.copy()

    # 同時更新 memory 中的 result_df
    if is_train:
        self.train_result_df = result_df
    else:
        self.test_result_df = result_df

    # 儲存 CSV
    # result_df.to_csv(file_path, index=False, mode = 'a')
    result_df.to_csv(file_path, index=False)
