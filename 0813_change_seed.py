import papermill as pm
import pandas as pd

summaries = []

import random

# 設定隨機種子，可以是一個數字或字串
random.seed(95)

# 創建一個空列表來儲存隨機數
seed_list = [95]

# 循環五次，每次生成一個隨機數並加入列表中
for _ in range(4):
    # random.randint(1, 100) 用來生成一個介於1到100之間的隨機整數
    random_number = random.randint(1, 100)
    seed_list.append(random_number)


for seed in seed_list:
    output_nb = f"./result/run_seed{seed}.ipynb"

    # 執行 notebook，傳入不同 seed
    pm.execute_notebook(
        '0812_stroke_analysis.ipynb',   # 原始 notebook 名字
        output_nb,               # 輸出 notebook，會包含執行結果
        parameters=dict(SEED=seed)
    )


import glob

def load_summary_any(path: str) -> pd.DataFrame:
    # 先嘗試用雙層表頭讀（因為 summary 有 ('AUC','mean') 這種欄位）
    df = pd.read_csv(path, header=[0,1])

    # 情況 1：第一欄本來就叫 ('model','') → 直接 set index
    if isinstance(df.columns, pd.MultiIndex) and ('model','') in df.columns:
        df = df.set_index(('model',''))

    # 情況 2：第一欄叫 'model'（單層） → 也行
    elif 'model' in df.columns:
        df = df.set_index('model')

    # 情況 3：第一欄是 Unnamed…（存檔時沒 index_label）→ 把它當成 model
    else:
        first_col = df.columns[0]   # 可能是 ('Unnamed: 0_level_0','Unnamed: 0_level_1') 或 'Unnamed: 0'
        df = df.set_index(first_col)
        # 讓 index 名稱好看一點
        try:
            df.index.name = 'model'
        except Exception:
            pass

    # 數字轉 float（避免字串）
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

# 讀多個 seeds
paths = sorted(glob.glob("./result/summary_seed*.csv"))
runs = []
for p in paths:
    one = load_summary_any(p)   # index = model, columns = MultiIndex[(metric, stat)]
    # 只拿每次 run 的「mean」欄位來做跨 seed 統計
    mean_cols = [c for c in one.columns if isinstance(c, tuple) and c[1]=='mean']
    one_mean = one[mean_cols].copy()
    # 攤平成單層欄名：('AUC','mean')->'AUC'，('thr','mean')->'thr'
    one_mean.columns = [m for (m,_) in one_mean.columns]
    runs.append(one_mean)

# 合併所有 run（index=model，columns=metrics）
res_all = pd.concat(runs, keys=range(len(runs)), names=['seed','model'])  # MultiIndex (seed, model)
# 跨 seed 計算 mean/std
summary_mean = res_all.groupby('model').mean(numeric_only=True).round(4)
summary_std  = res_all.groupby('model').std(numeric_only=True).round(4)

print("=== Mean across seeds (using per-run 'mean' values) ===")
print(summary_mean)
print("\n=== Std across seeds (using per-run 'mean' values) ===")
print(summary_std)

# 如果你想再組回原本那種「每個 metric 有 mean/std 兩欄」的寬表：
wide_out = pd.concat(
    {m: pd.concat({'mean': summary_mean[m], 'std': summary_std[m]}, axis=1)
     for m in summary_mean.columns},
    axis=1
).round(4)
print("\n=== Combined summary table (multi-index columns like original) ===")
print(wide_out)

# 存檔
summary_mean.to_csv("./result/summary_across_seeds_mean.csv", index=True)
summary_std.to_csv("./result/summary_across_seeds_std.csv", index=True)
wide_out.to_csv("./result/summary_across_seeds_wide.csv", index=True)
