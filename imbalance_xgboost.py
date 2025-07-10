import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report

# ==== Step 1: Load and prepare data ====
data = pd.read_csv('raw_data/age_below_66.csv')  # 替換為你的資料路徑

# 特徵與標籤欄位
selected_features = ['age', 'sex', 'HLOS', 'NIHSS', 'tPA(0/1)', 'EVT(0/1)',
                     'HTN(0/1)', 'DM(0/1)', 'Dyslipidemia(0/1)', 'Af(0/1)',
                     'smoking(Y/N/Q)', 'LDL ', 'cholesterol', 'TG', 'Cre', 'SGPT', 'HbA1c', 'MRS']
label = 'Second_Stroke'

X = data[selected_features]
y = data[label]

# 顯示正負樣本數量
print(f"✅ 正樣本（Second Stroke=1）數量: {sum(y==1)}")
print(f"✅ 負樣本（Second Stroke=0）數量: {sum(y==0)}")

# 分割訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)

# ==== Step 2: 標準化 ====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==== Step 3: 計算平衡權重 ====
# 計算正負樣本比例：負樣本數 / 正樣本數
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

# ==== Step 4: 訓練 Imbalanced-XGBoost ====
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    random_state=42
)

model.fit(X_train_scaled, y_train, 
          eval_metric=['auc', 'aucpr', 'logloss'],
          eval_set=[(X_test_scaled, y_test)],
          verbose=True)



# model.fit(X_train_scaled, y_train)

# ==== Step 5: 預測與評估 ====
# 預測機率
from sklearn.metrics import classification_report, confusion_matrix
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 調整閾值（預測為 1 的門檻）
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# 評估
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))