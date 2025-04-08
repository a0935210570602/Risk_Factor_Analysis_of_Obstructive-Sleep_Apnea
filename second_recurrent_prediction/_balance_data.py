# smote_augmenter.py
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
import pandas as pd

# 標準SMOTE方法
def smote_standard(self):
    self.balance_config["name"] = "smote_standard"
    random_state = self.balance_config.get("random_state", 42)
    k_neighbors = self.balance_config.get("neighbors", 5)

    # 印出原始標籤分布
    print("原始資料標籤分布：")
    print(self.train_Y.iloc[:, 0].value_counts())

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    self.train_X, self.train_Y = smote.fit_resample(self.train_X, self.train_Y)

    # 印出平衡後標籤分布
    print("SMOTE 平衡後標籤分布：")
    print(self.train_Y.iloc[:, 0].value_counts())
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
    print(self.train_Y.iloc[:, 0].value_counts())

    random_state = self.balance_config.get("random_state", 42)
    k_neighbors = self.balance_config.get("neighbors", 5)
    borderline_smote = BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors, kind=kind)
    self.train_X, self.train_Y = borderline_smote.fit_resample(self.train_X, self.train_Y)

    # 印出平衡後標籤分布
    print("SMOTE 平衡後標籤分布：")
    print(self.train_Y.iloc[:, 0].value_counts())
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
    svm_smote = SVMSMOTE(random_state=random_state, k_neighbors=k_neighbors)
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
    adasyn = ADASYN(random_state=random_state, n_neighbors=n_neighbors)
    self.train_X, self.train_Y =  adasyn.fit_resample(self.train_X, self.train_Y)
    self.balance_config["sample amount"] = {
        "positive (1)": int((self.train_Y == 1).sum()),
        "negative (0)": int((self.train_Y == 0).sum()),
        "total": int(len(self.train_Y))
    }