import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# ==== 1. 讀入資料 ====
data = pd.read_csv("raw_data/female_data.csv")
selected_features = ['age', 'sex', 'HLOS', 'NIHSS', 'tPA(0/1)', 'EVT(0/1)',
                     'HTN(0/1)', 'DM(0/1)', 'Dyslipidemia(0/1)', 'Af(0/1)',
                     'smoking(Y/N/Q)', 'LDL ', 'cholesterol', 'TG', 'Cre', 'SGPT', 'HbA1c', 'MRS']

X = data[selected_features]
y = data['Second_Stroke']

# ==== 2. 拆出正常與異常樣本 ====
X_normal = X[y == 0]
X_anomaly = X[y == 1]

# ==== 3. 標準化 ====
scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)
X_anomaly_scaled = scaler.transform(X_anomaly)

# ==== 4. 切分正常樣本為訓練/測試 ====
X_train, X_test_normal = train_test_split(X_normal_scaled, test_size=0.2, random_state=42)
X_test = np.vstack([X_test_normal, X_anomaly_scaled])
y_test = np.array([0]*len(X_test_normal) + [1]*len(X_anomaly_scaled))

# ==== 5. 建構 Autoencoder ====
# ==== 5. 建構改良版 Autoencoder ====
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
encoded = Dense(4, activation='relu')(encoded)         # 壓縮到 4 維
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(0.001), loss='mse')

# ==== 6. 訓練 Autoencoder ====
autoencoder.fit(X_train, X_train,
                epochs=150,
                batch_size=16,
                shuffle=True,
                validation_split=0.1,
                verbose=1)

# ==== 7. 計算重建誤差 ====
recon = autoencoder.predict(X_test)
recon_error = np.mean(np.square(X_test - recon), axis=1)

# ==== 8. 決定閾值（可用 training 的 95% 分位） ====
train_recon = autoencoder.predict(X_train)
train_error = np.mean(np.square(X_train - train_recon), axis=1)
threshold = np.percentile(train_error, 95)

# ==== 9. 根據重建誤差做異常判定 ====
y_pred = (recon_error > threshold).astype(int)

# ==== 10. 評估 ====
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

plt.hist(train_error, bins=50, alpha=0.6, label='Train Normal')
plt.hist(recon_error[y_test == 0], bins=50, alpha=0.6, label='Test Normal')
plt.hist(recon_error[y_test == 1], bins=50, alpha=0.6, label='Test Stroke')
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.title("Reconstruction Error Distribution")
plt.xlabel("MSE")
plt.ylabel("Count")
plt.show()
