import numpy as np
import pandas as pd

# 讀取資料
df = pd.read_csv('./raw_data/age_below_65.csv')

# 篩選 Second_Stroke == 0 的資料
df_non_stroke = df[df['Second_Stroke'] == 0].copy()

# 將資料等分成 5 份
split_dfs = np.array_split(df_non_stroke, 5)

# 儲存為 age_below_65_N1.csv ~ N5.csv
output_paths = []
for i, part in enumerate(split_dfs):
    file_path = f'./raw_data/age_below_65_N{i+1}.csv'
    part.to_csv(file_path, index=False)
    output_paths.append(file_path)

output_paths
