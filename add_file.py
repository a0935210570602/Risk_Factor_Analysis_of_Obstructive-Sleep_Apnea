import os

import numpy as np
import pandas as pd

# 重新載入非中風資料切分檔案
n_files = [pd.read_csv(f"./raw_data/age_below_65_N{i+1}.csv") for i in range(5)]

# 讀取 age_below_65.csv 原始檔案（這次我會改為你剛上傳的最新版本）
original_df = pd.read_csv("./raw_data/age_below_65.csv")

# 篩選 Second_Stroke == 1 的資料
df_stroke = original_df[original_df['Second_Stroke'] == 1].copy()

# 平均切成 5 份，分別加到 N1~N5 中
# stroke_splits = np.array_split(df_stroke, 5)
updated_n_files = [pd.concat([n_files[i], df_stroke], ignore_index=True) for i in range(5)]

# 儲存合併後的檔案
updated_paths = []
for i, df in enumerate(updated_n_files):
    path = f"./raw_data/age_below_65_N{i+1}_with_stroke.csv"
    df.to_csv(path, index=False)
    updated_paths.append(path)

updated_paths
