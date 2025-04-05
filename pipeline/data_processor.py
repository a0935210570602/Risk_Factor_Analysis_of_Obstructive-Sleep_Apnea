import pandas as pd
from sklearn import model_selection

SELECTED_FEATURE_LIST = ['age', 'sex', 'HLOS', 'NIHSS', 'tPA(0/1)', 'EVT(0/1)',
                          'HTN(0/1)', 'DM(0/1)', 'Dyslipidemia(0/1)', 'Af(0/1)',
                          'smoking(Y/N/Q)', 'LDL ', 'cholesterol', 'TG', 'Cre',
                          'SGPT', 'HbA1c', 'MRS']
LABEL_NAME = ['Second_Stroke']

class DataProcessor:
    def __init__(self, data_config):
        """
        初始化 DataProcessor

        Args:
            data_config (dict): 資料配置字典，包含以下鍵：
                - "path": 資料檔案路徑，例如 "raw_data/age_below_65.csv"
                - "test_size": 測試集比例，例如 0.2
        """
        self.path = data_config.get("path")
        self.test_size = data_config.get("test_size", 0.2)
        self.train_size = 1 - self.test_size
        self.random_state = data_config.get("random_state", 42)

    def load_data(self):
        """
        載入 CSV 數據，根據 "Second_Stroke" 欄位將資料分為兩組：
            - Stroke 資料（Second_Stroke == 1）
            - Normal 資料（Second_Stroke == 0）
        分別對兩組資料以 80%/20% 拆分為訓練集與測試集，
        最後將兩組訓練集與測試集合併，並依據 SELECTED_FEATURE_LIST 與 LABEL_NAME 選取特徵與標籤。

        Returns:
            tuple: (train_X, train_Y, test_X, test_Y)
        """
        # 讀取資料
        df = pd.read_csv(self.path)
        
        # 根據 "Second_Stroke" 欄位抓取資料
        stroke_df = df[df["Second_Stroke"] == 1]
        normal_df = df[df["Second_Stroke"] == 0]
        
        # 分別拆分正常與中風資料 (80% train, 20% test)
        normal_train_df, normal_test_df = model_selection.train_test_split(
            normal_df, train_size = self.train_size, random_state= self.random_state)
        stroke_train_df, stroke_test_df = model_selection.train_test_split(
            stroke_df, train_size=self.test_size, random_state=self.random_state)
        
        # 合併訓練集與測試集
        train_df = pd.concat([stroke_train_df, normal_train_df], axis=0).reset_index(drop=True)
        test_df = pd.concat([stroke_test_df, normal_test_df], axis=0).reset_index(drop=True)
        
        # 選取特徵與標籤
        train_X = train_df[SELECTED_FEATURE_LIST]
        train_Y = train_df[LABEL_NAME]
        test_X = test_df[SELECTED_FEATURE_LIST]
        test_Y = test_df[LABEL_NAME]
        
        return train_X, train_Y, test_X, test_Y
