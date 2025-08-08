import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, classification_report

# ==== Step 1: Load and prepare data ====
data = pd.read_csv('raw_data/female_data.csv')  # 替換成你的路徑

# 假設你想用下列欄位作為特徵
selected_features = ['age', 'sex', 'HLOS', 'NIHSS', 'tPA(0/1)', 'EVT(0/1)',
                     'HTN(0/1)', 'DM(0/1)', 'Dyslipidemia(0/1)', 'Af(0/1)',
                     'smoking(Y/N/Q)', 'LDL ', 'cholesterol', 'TG', 'Cre', 'SGPT', 'HbA1c', 'MRS']
label = 'Second_Stroke'

X = data[selected_features]
y = data[label]

# 只用正常樣本訓練 One-Class SVM（以 0 為正常）
X_normal = X[y == 0]
X_anomaly = X[y == 1]  # 作為後續測試異常

print(f"✅ 正樣本（Second Stroke=1）數量: {len(X_anomaly)}")
print(f"✅ 負樣本（Second Stroke=0）數量: {len(X_normal)}")


# 切分部分正常樣本為測試集，模擬真實場景
X_normal_train, X_normal_test = train_test_split(X_normal, test_size=0.1, random_state=42)

# 組合測試集（含部分正常樣本 + 所有異常樣本）
X_test = pd.concat([X_normal_test, X_anomaly], axis=0)
y_test = [0] * len(X_normal_test) + [1] * len(X_anomaly)

# ==== Step 2: 標準化 ====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_normal_train)
X_test_scaled = scaler.transform(X_test)

# ==== Step 3: 模型訓練 ====
# model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
model = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.2)  # 提高 nu 表示更敏感

model.fit(X_train_scaled)

# ==== Step 4: 預測與評估 ====
y_pred = model.predict(X_test_scaled)  # 結果為 1（正常）或 -1（異常）
y_pred_binary = [0 if p == 1 else 1 for p in y_pred]  # 轉為 0/1 與 y_test 對齊

# 結果輸出
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))
