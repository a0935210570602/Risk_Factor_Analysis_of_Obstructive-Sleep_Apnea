import gc
import pandas as pd
import os
from memory_profiler import profile
from sklearn import model_selection, metrics
from sklearn.discriminant_analysis import StandardScaler
from sklearn.utils import shuffle

def apply_standardization(self):
    self.standardization_or_not = True

def set_smote_method(self, smote_method):
    self.smote_method = smote_method

def set_feature_selection_method(self, feature_selection_method):
    self.feature_selection_method = feature_selection_method

def feature_selection_method(self):
    self.feature_selection_method(self)
    # pass

def smote_method(self):
    self.smote_method(self)

def set_downsampling_rate(self, rate):
    self.data_config["down_sampling_rate"] = rate

# 每次實驗結束後手動釋放資料框和模型物件
@profile
def clear_data_and_model(self):
    # 設為 None，幫助釋放大型物件的記憶體
    self.data_df = None
    self.train_X = None
    self.train_Y = None
    self.valid_X = None
    self.valid_Y = None
    self.data_X = None
    self.data_Y = None

    # 強制刪除變數參考
    del self.data_df
    del self.train_X
    del self.train_Y
    del self.valid_X
    del self.valid_Y
    del self.data_X
    del self.data_Y

    # 模型物件釋放（若存在）
    for model_attr in [
        'linear_svc_model', 'poly_svc_model', 'rbf_svc_model',
        'decision_tree_model', 'forest_model', 'xgboost_model',
        'adaboost_model', 'grad_boost_model'
    ]:
        if hasattr(self, model_attr):
            setattr(self, model_attr, None)
            delattr(self, model_attr)

    # 關閉圖形（如有畫圖）
    import matplotlib.pyplot as plt
    plt.close('all')

    # 多次強制垃圾回收
    for _ in range(3):
        gc.collect()

# TESTED
# def load_data(self):
#     # 逐一取出各個變數
#     data_path = self.data_config["path"]
#     train_size = self.data_config["train_size"]
#     random_state = self.data_config.get("random_state", 42)
    
#     self.data_df = pd.read_csv(data_path)

#     # Split data to Training set & Testing set
#     stroke_df = self.data_df[self.data_df["Second_Stroke"] == 1]
#     normal_df = self.data_df[self.data_df["Second_Stroke"] == 0]

#     # down_sampling_rate = self.data_config.get("down_sampling_rate", 1.0)
#     # if down_sampling_rate < 1.0:
#     #     normal_df = normal_df.sample(frac=down_sampling_rate, random_state=random_state)

#     self.normal_train_df, self.normal_test_df = model_selection.train_test_split(
#             normal_df, train_size=train_size, random_state=random_state, stratify=normal_df["Second_Stroke"])
#     self.stroke_train_df, self.stroke_test_df = model_selection.train_test_split(
#             stroke_df, train_size=train_size, random_state=random_state, stratify=stroke_df["Second_Stroke"])



#     train_df = pd.concat([self.stroke_train_df, self.normal_train_df], axis = 0)
#     test_df = pd.concat([self.stroke_test_df, self.normal_test_df], axis = 0)

#     # 將資料隨機打亂並重設索引
#     train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
#     test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
#     # Select features of training & testing data
#     self.train_X = train_df[self.SELECTED_FEATURE_LIST]
#     self.train_Y = train_df[self.LABEL_NAME].values.ravel()
    
#     self.valid_X = test_df[self.SELECTED_FEATURE_LIST]
#     self.valid_Y = test_df[self.LABEL_NAME].values.ravel()

#     self.data_X = self.data_df[self.SELECTED_FEATURE_LIST]
#     self.data_Y = self.data_df[self.LABEL_NAME].values.ravel()
#     # self.data_X = self.train_X.append(self.valid_X, ignore_index=True)
#     # self.data_Y = self.train_Y.tolist() + self.valid_Y.tolist()

#     # 印出原始標籤分布
#     print("原始資料標籤分布(train)：")
#     print(pd.Series(self.train_Y).value_counts())
#     print("原始資料標籤分布(test)：")
#     print(pd.Series(self.valid_Y).value_counts())

def load_data(self):
    data_path = self.data_config["path"]

    self.data_df = pd.read_csv(data_path)

    self.continuous_features = ['age', 'HLOS', 'NIHSS', 'LDL ', 'cholesterol', 'TG', 'Cre', 'SGPT', 'HbA1c']
    self.categorical_features = ['sex', 'tPA(0/1)', 'EVT(0/1)', 'HTN(0/1)', 'DM(0/1)', 
                                 'Dyslipidemia(0/1)', 'Af(0/1)', 'smoking(Y/N/Q)', 'MRS']

    # 合併 X 與 y
    self.data_X = self.data_df[self.continuous_features + self.categorical_features]
    self.data_Y = self.data_df["Second_Stroke"]

def prepare_tenfold_data(self):
    # 分成正負樣本
    stroke_df = self.data_df[self.data_df["Second_Stroke"] == 1]
    normal_df = self.data_df[self.data_df["Second_Stroke"] == 0]

    random_state = self.data_config.get("random_state", 42)

    # 建立 StratifiedKFold 物件
    skf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    # 將正負樣本合併並建立標籤
    combined_df = pd.concat([stroke_df, normal_df]).reset_index(drop=True)
    y = combined_df["Second_Stroke"].values  # 用來做 stratification

    # 儲存每一 fold 的索引 (train, val)
    self.fold_indices = []  # list of (train_idx, val_idx) tuples

    for train_index, val_index in skf.split(combined_df, y):
        self.fold_indices.append((train_index, val_index))

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pandas as pd

def standardize(self):
    """
    如果 continuous_features 或 categorical_features 列表中的某些欄位在 train_X/valid_X
    中不存在，就跳過那些欄位，只對剩下存在的欄位執行標準化/合併。
    """
    # 1. 如果物件沒有屬性 standardization_or_not 或其值為 False，就跳過
    if not getattr(self, "standardization_or_not", False):
        return

    # 2. 確認 train_X 和 valid_X 存在
    if not hasattr(self, "train_X") or not hasattr(self, "valid_X"):
        return

    # 3. 檢查 continuous_features 和 categorical_features 是否存在
    cont_feats = getattr(self, "continuous_features", [])
    cat_feats  = getattr(self, "categorical_features", [])

    # 4. 找出 train_X/valid_X 中實際存在的欄位
    existing_cont_train = [col for col in cont_feats if col in self.train_X.columns]
    existing_cont_val   = [col for col in cont_feats if col in self.valid_X.columns]
    existing_cat_train  = [col for col in cat_feats  if col in self.train_X.columns]
    existing_cat_val    = [col for col in cat_feats  if col in self.valid_X.columns]

    # 5. 如果連一個 continuous 欄位都沒有，就跳過標準化流程，只保留現有的 categorical_features
    if not existing_cont_train or not existing_cont_val:
        # 直接過濾 train_X 和 valid_X，只保留現有的 categorical_features
        self.train_X = self.train_X[existing_cat_train].copy()
        self.valid_X = self.valid_X[existing_cat_val].copy()
        return

    # 6. 執行標準化 (只對存在的 continuous 欄位做 fit/transform)
    scaler = StandardScaler()
    try:
        # fit_transform train 的現有 continuous 欄位
        X_train_cont_scaled = scaler.fit_transform(self.train_X[existing_cont_train])
        # transform valid 的現有 continuous 欄位
        X_val_cont_scaled   = scaler.transform(self.valid_X[existing_cont_val])

        # 7. 把標準化後的 continuous 欄位轉回 DataFrame，並與存在的 categorical 欄位合併
        df_train_cont = pd.DataFrame(
            X_train_cont_scaled,
            columns=existing_cont_train,
            index=self.train_X.index
        )
        df_val_cont = pd.DataFrame(
            X_val_cont_scaled,
            columns=existing_cont_val,
            index=self.valid_X.index
        )

        # 8. 如果 categorical 欄位存在，就一併合併；否則只回傳 continuous
        if existing_cat_train:
            self.train_X = pd.concat(
                [df_train_cont, self.train_X[existing_cat_train]],
                axis=1
            )
        else:
            self.train_X = df_train_cont

        if existing_cat_val:
            self.valid_X = pd.concat(
                [df_val_cont, self.valid_X[existing_cat_val]],
                axis=1
            )
        else:
            self.valid_X = df_val_cont

    except Exception:
        # 如果在標準化過程中發生任何錯誤，就跳過、不丟例外
        return


def dowmsample(self):
    from sklearn.utils import shuffle

    # 取出正負樣本
    X_pos = self.train_X[self.train_Y == 1]
    y_pos = self.train_Y[self.train_Y == 1]
    X_neg = self.train_X[self.train_Y == 0]
    y_neg = self.train_Y[self.train_Y == 0]

    # 計算要保留多少負樣本數量
    target_neg_count = int(len(y_pos) * 1.5)

    # 隨機抽樣負樣本
    X_neg_sampled = X_neg.sample(n=target_neg_count, random_state=42)
    y_neg_sampled = y_neg.loc[X_neg_sampled.index]

    # 合併正樣本與抽樣後的負樣本
    self.train_X = pd.concat([X_pos, X_neg_sampled], axis=0)
    self.train_Y = pd.concat([y_pos, y_neg_sampled], axis=0)

    # 打亂
    self.train_X, self.train_Y = shuffle(self.train_X, self.train_Y, random_state=42)

    # 印出結果
    print("✅ Downsample 後標籤分布：")
    print(pd.Series(self.train_Y).value_counts())

def cross_validation(self):
    results = []  # 儲存每 fold 的結果

    for fold_id, (train_idx, val_idx) in enumerate(self.fold_indices):
        print(f"🔁 Fold {fold_id + 1}/10")

        # 抓出資料
        self.train_X = self.data_X.iloc[train_idx].copy()
        self.train_Y = self.data_Y.iloc[train_idx].copy()
        self.valid_X= self.data_X.iloc[val_idx].copy()
        self.valid_Y = self.data_Y.iloc[val_idx].copy()
        # 🔍 列印資料筆數
        print(f"📊 Fold {fold_id + 1} - Train: {len(self.train_X)} samples, Valid: {len(self.valid_X)} samples")
        self.fold = fold_id + 1  # 設定當前 fold 編號
        standardize(self)
        smote_method(self)
        dowmsample(self)
        feature_selection_method(self)

        # 訓練與預測
        self.prediction_method(self)
    # append_average_to_existing(self)

import os
import pandas as pd

def append_average_to_existing(self):
    """
    讀取已有的 train_<filename>.csv / test_<filename>.csv（裡面已有 10 折每折的紀錄），
    計算每個 Model name 的平均值，然後把平均值那一列 append 到原檔案底下並覆寫。
    """

    # 1. 先把資料檔名（例如 age_below_65.csv）拆出主檔名
    filename = os.path.basename(self.data_config["path"])
    name_without_ext, _ = os.path.splitext(filename)

    # 2. 處理 train 的檔案
    train_dir  = os.path.join(self.PATH, "train")
    train_path = os.path.join(train_dir, f"train_{name_without_ext}.csv")
    if os.path.exists(train_path):
        # 2.1 讀取原本 10 折的 train 紀錄（10 行 + 可能其他 model）
        df_train = pd.read_csv(train_path)

        # 2.2 計算每個 Model name 在 10 折上的平均
        agg_cols = ["accuracy", "precision", "recall", "f1‐score", "auc"]
        # groupby 後針對上面欄位做 mean
        df_avg = (
            df_train
            .groupby("Model name")[agg_cols]
            .mean()
            .round(2)
            .reset_index()
        )

        # 2.3 取出每個 model 對應的「其它欄位」（feature、balance_config、downsampling_rate、feature_selection_config）來當輔助，
        #     用 groupby.first() 把這些欄位的第一筆值抓出來（因為同一個 model 在每折的這些欄位其實都一樣）
        extras = (
            df_train
            .groupby("Model name")[["feature", "balance_config", "downsampling_rate", "feature_selection_config"]]
            .first()
            .reset_index()
        )

        # 2.4 把 extras 與 df_avg merge 起來，並加上 dataset="train"、Fold="avg"
        df_avg = pd.merge(extras, df_avg, on="Model name", how="left")
        df_avg["dataset"] = "train"
        df_avg["Fold"]    = "avg"  # 或 "average"

        # 2.5 重新排列欄位順序，對應到原本 df_train 的順序：
        #     ['Model name', 'Fold', 'dataset', 'accuracy', 'precision', 'recall', 'f1‐score', 'auc',
        #      'feature', 'balance_config', 'downsampling_rate', 'feature_selection_config']
        cols_order = [
            "Model name", "Fold", "dataset",
            "accuracy", "precision", "recall", "f1‐score", "auc",
            "feature", "balance_config", "downsampling_rate", "feature_selection_config"
        ]
        df_avg = df_avg[cols_order]

        # 2.6 把原本 df_train 和 df_avg 合併，再覆寫回 train_path
        df_combined = pd.concat([df_train, df_avg], ignore_index=True)
        df_combined.to_csv(train_path, index=False)
        print(f"已把 train 平均值追加到：{train_path}")
    else:
        print(f"[警告] 找不到 train 檔案：{train_path}，無法追加平均值。")

    # 3. 處理 test 的檔案（步驟與 train 幾乎一樣）
    test_dir  = os.path.join(self.PATH, "test")
    test_path = os.path.join(test_dir, f"test_{name_without_ext}.csv")
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)

        agg_cols = ["accuracy", "precision", "recall", "f1‐score", "auc"]
        df_avg_test = (
            df_test
            .groupby("Model name")[agg_cols]
            .mean()
            .round(2)
            .reset_index()
        )
        extras_test = (
            df_test
            .groupby("Model name")[["feature", "balance_config", "downsampling_rate", "feature_selection_config"]]
            .first()
            .reset_index()
        )

        df_avg_test = pd.merge(extras_test, df_avg_test, on="Model name", how="left")
        df_avg_test["dataset"] = "test"
        df_avg_test["Fold"]    = "avg"
        cols_order = [
            "Model name", "Fold", "dataset",
            "accuracy", "precision", "recall", "f1‐score", "auc",
            "feature", "balance_config", "downsampling_rate", "feature_selection_config"
        ]
        df_avg_test = df_avg_test[cols_order]

        df_combined_test = pd.concat([df_test, df_avg_test], ignore_index=True)
        df_combined_test.to_csv(test_path, index=False)
        print(f"已把 test 平均值追加到：{test_path}")
    else:
        print(f"[警告] 找不到 test 檔案：{test_path}，無法追加平均值。")

    # # 合併所有 fold 的結果
    # self.cross_val_results = pd.DataFrame(results)



# TODO
# 10-fold
# def cross_validation(self, model, name):
#     from imblearn.pipeline import Pipeline
#     kind='borderline-1'
#     from imblearn.over_sampling import BorderlineSMOTE

#     smote = BorderlineSMOTE(random_state=42, k_neighbors=5, kind=kind)

#     # 包裝成管線：先 SMOTE 後訓練模型
#     pipeline = Pipeline([
#         ("smote", smote),
#         ("clf", model)
#     ])

#     scores = model_selection.cross_validate(
#         pipeline, self.data_X, self.data_Y,
#         cv=5,
#         scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'),
#     )

#     # Convert the scores to a DataFrame
#     df_score = pd.DataFrame(scores)

#     # Calculate mean and standard deviation of the scores
#     avg_scores = df_score.mean()
#     std_scores = df_score.std()

#     # Save the scores of each fold to a separate CSV file (overwrite if exists)
#     dir_path = os.path.join(self.PATH, '5-fold')
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)
#     file_path = os.path.join(dir_path, f'{name}_5-fold_scores.csv')
#     df_score.to_csv(file_path, index=False)

#     # Generate DataFrame for average scores and standard deviations
#     avg_result_df = pd.DataFrame({
#         'Model name': [name],
#         'dataset': ['5-fold'],
#         'accuracy_mean': [avg_scores['test_accuracy']],
#         'accuracy_std': [std_scores['test_accuracy']],
#         'precision_mean': [avg_scores['test_precision']],
#         'precision_std': [std_scores['test_precision']],
#         'recall_mean': [avg_scores['test_recall']],
#         'recall_std': [std_scores['test_recall']],
#         'f1-score_mean': [avg_scores['test_f1']],
#         'f1-score_std': [std_scores['test_f1']],
#         'down_sampling_rate':[self.data_config["down_sampling_rate"]],
#         'auc_mean': [avg_scores['test_roc_auc']],
#         'auc_std': [std_scores['test_roc_auc']],
#         'feature':[self.SELECTED_FEATURE_LIST],
#         'balance_config':[self.balance_config],
#         'feature_selection_config':[self.feature_selection_config],
#     })

#     dir_path = os.path.join(self.PATH, '5-fold')
#     file_path = os.path.join(dir_path, 'all_models_5-fold-avg-std.csv')
#     if not os.path.isdir(dir_path):  # 確認儲存檔案位置 若沒有的話 則新建檔案
#         os.makedirs(dir_path)
#     # Update the global DataFrame (remove any previous entry for the same model)
#     if os.path.exists(file_path):
#         # print(self.ten_fold_avg_std_df.head(5))
#         # print(avg_result_df.head(5))
#         existing_df = pd.read_csv(file_path)
#         self.ten_fold_avg_std_df = pd.concat([existing_df, avg_result_df], ignore_index=True)
#         # self.ten_fold_avg_std_df = self.ten_fold_avg_std_df[self.ten_fold_avg_std_df['Model name'] != name]
#         # self.ten_fold_avg_std_df = pd.concat([self.ten_fold_avg_std_df, avg_result_df], axis=0, ignore_index=True)
#     else:
#         self.ten_fold_avg_std_df = avg_result_df.copy()


#     # Save the average and standard deviation summary (overwrite if exists)
#     self.ten_fold_avg_std_df.to_csv(file_path, index=False)

def display_total_result_train(self):
    print(self.train_result_df)

def display_total_result_test(self):
    print(self.test_result_df)

def display_total_ten_fold_result(self):
    print(self.ten_fold_avg_std_df)