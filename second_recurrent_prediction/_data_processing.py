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
    self.X = self.data_df[self.continuous_features + self.categorical_features]
    self.y = self.data_df["Second_Stroke"]

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
    if self.standardization_or_not:
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.train_X[self.continuous_features])
        self.X_val_scaled = scaler.transform(self.valid_X[self.continuous_features])

        # 合併類別特徵
        self.train_X= pd.concat([
            pd.DataFrame(self.X_train_scaled, columns=self.continuous_features, index=self.train_X.index),
            self.train_X[self.categorical_features]
        ], axis=1)
        self.valid_X= pd.concat([
            pd.DataFrame(self.X_val_scaled, columns=self.continuous_features, index=self.valid_X.index),
            self.valid_X[self.categorical_features]
        ], axis=1)
        
def cross_validation(self):
    results = []  # 儲存每 fold 的結果

    for fold_id, (train_idx, val_idx) in enumerate(self.fold_indices):
        print(f"🔁 Fold {fold_id + 1}/10")

        # 抓出資料
        self.train_X = self.X.iloc[train_idx]
        self.train_Y = self.y.iloc[train_idx]
        self.valid_X= self.X.iloc[val_idx]
        self.valid_Y = self.y.iloc[val_idx]
        # 🔍 列印資料筆數
        print(f"📊 Fold {fold_id + 1} - Train: {len(self.train_X)} samples, Valid: {len(self.valid_X)} samples")
        
        smote_method(self)
        standardize(self)

        # 訓練與預測
        self.prediction_method(self)

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