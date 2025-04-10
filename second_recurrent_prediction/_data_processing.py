import gc
import pandas as pd
import os
from memory_profiler import profile
from sklearn import model_selection, metrics
from sklearn.discriminant_analysis import StandardScaler

def set_downsampling_rate(self, rate):
    self.data_config["down_sampling_rate"] = rate

# 每次實驗結束後手動釋放資料框和模型物件
@profile
def clear_data_and_model(self):
    # 設為 None，幫助釋放大型物件的記憶體
    self.data_df = None
    self.train_X = None
    self.train_Y = None
    self.test_X = None
    self.test_Y = None
    self.data_X = None
    self.data_Y = None

    # 強制刪除變數參考
    del self.data_df
    del self.train_X
    del self.train_Y
    del self.test_X
    del self.test_Y
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
def load_data(self):
    self.SELECTED_FEATURE_LIST = ['age', 'sex', 'HLOS', 'NIHSS', 'tPA(0/1)', 'EVT(0/1)',
        'HTN(0/1)', 'DM(0/1)', 'Dyslipidemia(0/1)', 'Af(0/1)', 'smoking(Y/N/Q)',
        'LDL ', 'cholesterol', 'TG', 'Cre', 'SGPT', 'HbA1c', 'MRS']
    # 逐一取出各個變數
    data_path = self.data_config["path"]
    train_size = self.data_config["train_size"]
    random_state = self.data_config["random_state"]
    
    self.data_df = pd.read_csv(data_path)

    # Split data to Training set & Testing set
    stroke_df = self.data_df[self.data_df["Second_Stroke"] == 1]
    normal_df = self.data_df[self.data_df["Second_Stroke"] == 0]

    down_sampling_rate = self.data_config.get("down_sampling_rate", 1.0)
    if down_sampling_rate < 1.0:
        stroke_df = stroke_df.sample(frac=down_sampling_rate, random_state=random_state)

    self.normal_train_df, self.normal_test_df = model_selection.train_test_split(
            normal_df, train_size=train_size, random_state=random_state)
    self.stroke_train_df, self.stroke_test_df = model_selection.train_test_split(
            stroke_df, train_size=train_size, random_state=random_state)

    train_df = pd.concat([self.stroke_train_df, self.normal_train_df], axis = 0)
    test_df = pd.concat([self.stroke_test_df, self.normal_test_df], axis = 0)

    # 將資料隨機打亂並重設索引
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Select features of training & testing data
    self.train_X = train_df[self.SELECTED_FEATURE_LIST]
    self.train_Y = train_df[self.LABEL_NAME].values.ravel()

    self.test_X = test_df[self.SELECTED_FEATURE_LIST]
    self.test_Y = test_df[self.LABEL_NAME].values.ravel()

    self.data_X = self.data_df[self.SELECTED_FEATURE_LIST]
    self.data_Y = self.data_df[self.LABEL_NAME].values.ravel()

def standardize_data(self):
    # 資料標準化：先對訓練資料 fit_transform，再 transform 測試及全體資料
    scaler = StandardScaler()
    self.train_X = pd.DataFrame(scaler.fit_transform(self.train_X),
                                columns=self.SELECTED_FEATURE_LIST)
    self.test_X = pd.DataFrame(scaler.transform(self.test_X),
                                columns=self.SELECTED_FEATURE_LIST)
    self.data_X = pd.DataFrame(scaler.transform(self.data_X),
                                columns=self.SELECTED_FEATURE_LIST)

# TODO
# 10-fold
def cross_validation(self, model, name):
    global ten_fold_avg_std_df


    # Perform 10-fold cross-validation
    scores = model_selection.cross_validate(model, self.data_X, self.data_Y, cv=10,
                            scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'),
                            return_train_score=True)

    # Convert the scores to a DataFrame
    df_score = pd.DataFrame(scores)

    # Calculate mean and standard deviation of the scores
    avg_scores = df_score.mean()
    std_scores = df_score.std()

    # Save the scores of each fold to a separate CSV file (overwrite if exists)
    dir_path = os.path.join(self.PATH, '10-fold')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, f'{name}_10-fold_scores.csv')
    df_score.to_csv(file_path, index=False)

    # Generate DataFrame for average scores and standard deviations
    avg_result_df = pd.DataFrame({
        'Model name': [name],
        'dataset': ['10-fold'],
        'accuracy_mean': [avg_scores['test_accuracy']],
        'accuracy_std': [std_scores['test_accuracy']],
        'precision_mean': [avg_scores['test_precision']],
        'precision_std': [std_scores['test_precision']],
        'recall_mean': [avg_scores['test_recall']],
        'recall_std': [std_scores['test_recall']],
        'f1-score_mean': [avg_scores['test_f1']],
        'f1-score_std': [std_scores['test_f1']],
        'auc_mean': [avg_scores['test_roc_auc']],
        'auc_std': [std_scores['test_roc_auc']],
        'feature':[self.SELECTED_FEATURE_LIST],
        'balance_config':[self.balance_config],
        'feature_selection_config':[self.feature_selection_config],
    })

    dir_path = os.path.join(self.PATH, '10-fold')
    file_path = os.path.join(dir_path, 'all_models_10-fold-avg-std.csv')
    if not os.path.isdir(dir_path):  # 確認儲存檔案位置 若沒有的話 則新建檔案
        os.makedirs(dir_path)
    # Update the global DataFrame (remove any previous entry for the same model)
    if not self.ten_fold_avg_std_df.empty:
        # print(self.ten_fold_avg_std_df.head(5))
        # print(avg_result_df.head(5))
        self.ten_fold_avg_std_df = self.ten_fold_avg_std_df[self.ten_fold_avg_std_df['Model name'] != name]
        self.ten_fold_avg_std_df = pd.concat([self.ten_fold_avg_std_df, avg_result_df], axis=0, ignore_index=True)
    else:
        self.ten_fold_avg_std_df = avg_result_df.copy()

    # Save the average and standard deviation summary (overwrite if exists)
    self.ten_fold_avg_std_df.to_csv(file_path, index=False)

def display_total_result_train(self):
    print(self.train_result_df)

def display_total_result_test(self):
    print(self.test_result_df)

def display_total_ten_fold_result(self):
    print(self.ten_fold_avg_std_df)