import pandas as pd
from sklearn import model_selection

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
        self.selected_feature_list = data_config.get("selected_feature_list", None)

    def load_data(self):
        """
        載入 CSV 數據，根據 "Second_Stroke" 欄位將資料分為兩組：
            - Stroke 資料（Second_Stroke == 1）
            - Normal 資料（Second_Stroke == 0）
        分別對兩組資料以 80%/20% 拆分為訓練集與測試集，
        並取得除了 "Second_Stroke" 外所有欄位作為特徵清單。

        Returns:
            tuple: (train_X, train_Y, test_X, test_Y)
        """
        # 讀取資料
        df = pd.read_csv(self.path)
        
        # 取得所有欄位（作為特徵）除了 "Second_Stroke"
        # feature_list = [col for col in df.columns if col != 'Second_Stroke']
        
        # 根據 "Second_Stroke" 欄位抓取資料
        stroke_df = df[df["Second_Stroke"] == 1]
        normal_df = df[df["Second_Stroke"] == 0]

        print(f"Stroke data count: {len(stroke_df)}")
        print(f"Normal data count: {len(normal_df)}")

        # 分別拆分正常與中風資料 (80% train, 20% test)
        normal_train_df, normal_test_df = model_selection.train_test_split(
            normal_df, train_size=self.train_size, random_state=self.random_state)
        stroke_train_df, stroke_test_df = model_selection.train_test_split(
            stroke_df, train_size=self.train_size, random_state=self.random_state)
        print("stroke_train_df: ", stroke_train_df.shape)
        print("stroke_test_df: ", stroke_test_df.shape)
        # normal_train_df, normal_test_df = model_selection.train_test_split(
        #     normal_df, train_size=self.train_size, random_state=self.random_state)
        # stroke_train_df, stroke_test_df = model_selection.train_test_split(
        #     stroke_df, train_size=self.test_size, random_state=self.random_state)
        
        # 合併訓練集與測試集
        train_df = pd.concat([stroke_train_df, normal_train_df], axis=0).reset_index(drop=True)
        test_df = pd.concat([stroke_test_df, normal_test_df], axis=0).reset_index(drop=True)
        

        # 利用 feature_list 選取特徵，並以 LABEL_NAME 作為標籤
        train_X = train_df[self.selected_feature_list]
        train_Y = train_df[LABEL_NAME]
        test_X = test_df[self.selected_feature_list]
        test_Y = test_df[LABEL_NAME]
        
        return train_X, train_Y, test_X, test_Y
