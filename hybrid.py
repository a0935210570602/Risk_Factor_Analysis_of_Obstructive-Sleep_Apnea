import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier

# === 1. 讀取資料與預處理 ===
data = pd.read_csv("raw_data/female_data.csv")

features = ['age', 'sex', 'HLOS', 'NIHSS', 'tPA(0/1)', 'EVT(0/1)',
            'HTN(0/1)', 'DM(0/1)', 'Dyslipidemia(0/1)', 'Af(0/1)',
            'smoking(Y/N/Q)', 'LDL ', 'cholesterol', 'TG', 'Cre', 'SGPT', 'HbA1c', 'MRS']
X = data[features]
y = data['Second_Stroke']

# === 2. 分割資料 ===
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# === 3. 建立 Autoencoder（只用正常樣本訓練） ===
X_train_normal = X_train_scaled[y_train == 0]
input_dim = X_train_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
encoded = Dense(4, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(0.001), loss='mse')
autoencoder.fit(X_train_normal, X_train_normal, epochs=200, batch_size=16, verbose=1, validation_split=0.1)

# === 4. 計算重建誤差並加入特徵 ===
train_recon = autoencoder.predict(X_train_scaled)
test_recon = autoencoder.predict(X_test_scaled)
re_train_err = np.mean(np.square(X_train_scaled - train_recon), axis=1).reshape(-1, 1)
re_test_err = np.mean(np.square(X_test_scaled - test_recon), axis=1).reshape(-1, 1)
X_train_hybrid = np.hstack([X_train_scaled, re_train_err])
X_test_hybrid = np.hstack([X_test_scaled, re_test_err])

# === 5. 用 XGBoost 預測 ===
model = XGBClassifier(scale_pos_weight=sum(y_train==0)/sum(y_train==1),
                      use_label_encoder=False,
                      eval_metric='aucpr',
                      random_state=42)
model.fit(X_train_hybrid, y_train)
y_pred = model.predict(X_test_hybrid)

# === 6. 評估 ===
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
