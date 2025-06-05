# smote_augmenter.py
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
import pandas as pd

# SMOTENC: 支援類別型特徵的 SMOTE（適合混合數值與類別資料）
from imblearn.over_sampling import SMOTENC

def smote_smotenc(self):
    self.balance_config["name"] = "smote_smotenc"
    random_state = self.balance_config.get("random_state", 42)
    k_neighbors = self.balance_config.get("neighbors", 5)

    # 根據你的 train_X 中的欄位，手動指定哪些是類別型欄位索引
    # 假設 sex(0/1)、HTN、DM 等為類別變數
    target_categorical_names = [
        'sex',
        'tPA(0/1)',
        'EVT(0/1)',
        'HTN(0/1)',
        'DM(0/1)',
        'Dyslipidemia(0/1)',
        'Af(0/1)',
        'smoking(Y/N/Q)',
        'MRS'
    ]

    categorical_features = [
        idx for idx, col in enumerate(self.train_X.columns)
        if col in target_categorical_names
    ]

    # 印出原始標籤分布
    print("原始資料標籤分布：")
    print(pd.Series(self.train_Y).value_counts())
    positive_count = (self.train_Y == 1).sum()
    target_count = positive_count*3
    smotenc = SMOTENC(
        sampling_strategy={1: target_count},
        categorical_features=categorical_features,
        random_state=random_state,
        k_neighbors=k_neighbors
    )

    original_len = len(self.train_Y)

    # 執行 SMOTENC
    self.train_X, self.train_Y = smotenc.fit_resample(self.train_X, self.train_Y)

    # 印出平衡後標籤分布
    print("SMOTENC 平衡後標籤分布：")
    print(pd.Series(self.train_Y).value_counts())

    # 印出其中一筆合成後的樣本（例如第一筆新增樣本）
    print(6666666)
    print("✅ 合成樣本第 1 筆（位置：第 {} 筆）如下：".format(original_len))
    print(self.train_X.iloc[original_len])
    print("Label:", self.train_X.iloc[original_len-1])
    self.balance_config["sample amount"] = {
        "positive (1)": int((self.train_Y == 1).sum()),
        "negative (0)": int((self.train_Y == 0).sum()),
        "total": int(len(self.train_Y))
    }


# 標準SMOTE方法
def smote_standard(self):
    self.balance_config["name"] = "smote_standard"
    random_state = self.balance_config.get("random_state", 42)
    k_neighbors = self.balance_config.get("neighbors", 5)

    # 印出原始標籤分布
    print("原始資料標籤分布：")
    print(pd.Series(self.train_Y).value_counts())
    positive_count = (self.train_Y == 1).sum()
    target_count = positive_count*3
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors,sampling_strategy={1: target_count})
    self.train_X, self.train_Y = smote.fit_resample(self.train_X, self.train_Y)

    # 印出平衡後標籤分布
    print("SMOTE 平衡後標籤分布：")
    print(pd.Series(self.train_Y).value_counts())

    self.balance_config["sample amount"] = {
        "positive (1)": int((self.train_Y == 1).sum()),
        "negative (0)": int((self.train_Y == 0).sum()),
        "total": int(len(self.train_Y))
    }
    

# Borderline SMOTE: 著重處理邊界附近樣本（分類邊界附近）
def smote_borderline(self):
    self.balance_config["name"] = "smote_borderline"
    kind='borderline-1'
    # 印出原始標籤分布
    print("原始資料標籤分布：")
    print(pd.Series(self.train_Y).value_counts())

    random_state = self.balance_config.get("random_state", 42)
    k_neighbors = self.balance_config.get("neighbors", 5)
    positive_count = (self.train_Y == 1).sum()
    target_count = positive_count*3
    borderline_smote = BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors, kind=kind,sampling_strategy={1: target_count})
    self.train_X, self.train_Y = borderline_smote.fit_resample(self.train_X, self.train_Y)

    # 印出平衡後標籤分布
    print("SMOTE 平衡後標籤分布：")
    print(pd.Series(self.train_Y).value_counts())
    self.balance_config["sample amount"] = {
        "positive (1)": int((self.train_Y == 1).sum()),
        "negative (0)": int((self.train_Y == 0).sum()),
        "total": int(len(self.train_Y))
    }

# SVM SMOTE: 使用支援向量機 (SVM) 決定生成新樣本的位置
def smote_svm(self):
    self.balance_config["name"] = "smote_svm"
    random_state = self.balance_config.get("random_state", 42)
    k_neighbors = self.balance_config.get("neighbors", 5)
    positive_count = (self.train_Y == 1).sum()
    target_count = positive_count*3
    svm_smote = SVMSMOTE(random_state=random_state, k_neighbors=k_neighbors,sampling_strategy={1: target_count})
    self.train_X, self.train_Y = svm_smote.fit_resample(self.train_X, self.train_Y)
    self.balance_config["sample amount"] = {
        "positive (1)": int((self.train_Y == 1).sum()),
        "negative (0)": int((self.train_Y == 0).sum()),
        "total": int(len(self.train_Y))
    }

# ADASYN: 自適應生成更難分類的樣本（錯分率高的區域樣本更多）
def smote_adasyn(self):
    self.balance_config["name"] = "smote_adasyn"
    random_state = self.balance_config.get("random_state", 42)
    n_neighbors = self.balance_config.get("neighbors", 5)
    positive_count = (self.train_Y == 1).sum()
    target_count = positive_count*3
    adasyn = ADASYN(random_state=random_state, n_neighbors=n_neighbors,sampling_strategy={1: target_count})
    self.train_X, self.train_Y =  adasyn.fit_resample(self.train_X, self.train_Y)
    self.balance_config["sample amount"] = {
        "positive (1)": int((self.train_Y == 1).sum()),
        "negative (0)": int((self.train_Y == 0).sum()),
        "total": int(len(self.train_Y))
    }