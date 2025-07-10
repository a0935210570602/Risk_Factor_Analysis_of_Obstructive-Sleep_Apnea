import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer, recall_score

import csv
from sklearn.model_selection import train_test_split
from sklearn import  preprocessing, metrics
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import cross_validate


def k_means_fit(self):
    '''
    n_cluster 分群類別數
    random_state隨機初始化, 選擇效果最好的一種來作為模型
    max_iter: 迭代次數
    '''
    kmeansModel = KMeans(n_clusters=2, random_state=46)
    clusters_pred = kmeansModel.fit_predict(self.train_X)
    #看各群集的中心點
    print("中心點\n",kmeansModel.cluster_centers_)

    #評估分群結果 (越大代表越差)
    print("分群結果:",kmeansModel.inertia_)
    
    #使用 silhouette scores 做模型評估
    # k = 1~9 做9次kmeans, 並將每次結果的inertia收集在一個list裡
    kmeans_list = [KMeans(n_clusters=k, random_state=46).fit(self.train_X)
                    for k in range(1, 10)]
    inertias = [model.inertia_ for model in kmeans_list]

    silhouette_scores = [silhouette_score(self.train_X, model.labels_)
                        for model in kmeans_list[1:]]

def svm_linear_fit(self):
    '''
    四種不同SVC核函數:

    kernel='linear' (線性)

    kernel='poly' (非線性)

    kernel='rbf' (非線性)

    kernel='sigmoid' (非線性)

    C: 限制模型的複雜度, 防止過度擬合。

    max_iter: 最大迭代次數, 預設1000。
    '''
    # 建立 linearSvc 模型
    self.linear_svc_model = svm.SVC(C=0.5, max_iter=3000,kernel='linear',probability=True,)
    # 使用訓練資料訓練模型
    self.linear_svc_model.fit(self.train_X, self.train_Y)
    # 使用訓練資料預測分類
    self.linear_train_predicted=self.linear_svc_model.predict(self.train_X)
    self.linear_valid_predicted=self.linear_svc_model.predict(self.valid_X)

    # 使用訓練資料預測機率
    self.linear_train_predicted_prob =self.linear_svc_model.predict_proba(self.train_X)
    self.linear_valid_predicted_prob =self.linear_svc_model.predict_proba(self.valid_X)

    # 計算準確率
    print('linear訓練集: ',self.linear_svc_model.score(self.train_X,self.train_Y))
    print('linear測試集: ',self.linear_svc_model.score(self.valid_X,self.valid_Y))
    print('========================')

# TODO: 這個模型還沒寫完
def dcnn_fit(self):
    import tensorflow as tf
    from tensorflow import keras
    
    import numpy as np
    X_train_balanced_reshaped = np.reshape(self.train_X.values, (-1, self.train_X.shape[1], 1))
    models = keras.models
    layers = keras.layers
    # 建立 DCNN 模型
    model = models.Sequential()
    model.add(layers.Input(shape=(self.train_X.shape[1], 1)))

    for _ in range(10):
        model.add(layers.Conv1D(filters=32, kernel_size=4, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(15, activation='sigmoid'))  # 改用 relu 試試
    model.add(layers.Dense(1, activation='sigmoid'))
    self.dcnn_model = model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    self.dcnn_model.compile(optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    from keras import Sequential
    class BalancedBatch(Sequential):
        def __init__(self, X, y, batch_size=10):
            self.X = X
    #         self.y = y.reset_index(drop=True)  # 確保 index 是連續的
            self.y = pd.Series(y)  # 轉換為 pandas Series

            self.batch_size = batch_size
            self.pos_indices = self.y[self.y == 1].index.to_numpy()
            self.neg_indices = self.y[self.y == 0].index.to_numpy()
            self.num_batches = min(len(self.pos_indices), len(self.neg_indices)) * 2 // batch_size
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)

        def __len__(self):
            return self.num_batches

        def __getitem__(self, idx):
            half = self.batch_size // 2
            start = idx * half
            end = start + half

            pos_idx = self.pos_indices[start:end]
            neg_idx = self.neg_indices[start:end]

            # 混合正負樣本
            batch_indices = np.concatenate([pos_idx, neg_idx])
            np.random.shuffle(batch_indices)

            X_batch = self.X[batch_indices]
            y_batch = self.y.iloc[batch_indices].to_numpy().reshape(-1, 1)

            # print("✅ Batch y 分布:", np.bincount(y_batch.flatten()))
            return X_batch, y_batch

    y_train_balanced = self.train_Y.copy()
    gen = BalancedBatch(X_train_balanced_reshaped, y_train_balanced, batch_size=10)

    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train_balanced_reshaped, y_train_balanced, 
        test_size=0.2, stratify=y_train_balanced, random_state=42)

    # 模型改用 gen 作為資料輸入
    self.dcnn_model.fit(
        gen,
        epochs=400,
    #     validation_data=(X_val, y_val.to_numpy().reshape(-1, 1)),  # 這裡補上 validation 資料
        validation_data=(X_val, y_val.reshape(-1, 1)),  # 這裡補上 validation 資料
        verbose=1
    )

    # 使用訓練資料預測分類
    self.dcnn_train_predicted = self.dcnn_model.predict(self.train_X)
    self.dcnn_test_predicted = self.dcnn_model.predict(self.valid_X)

    self.dcnn_train_predicted_prob = self.dcnn_model.predict_proba(self.train_X)
    self.dcnn_test_predicted_prob = self.dcnn_model.predict_proba(self.valid_X)

    # 計算準確率
    print('訓練集: ',self.dcnn_model.score(self.train_X,self.train_Y))
    print('測試集: ',self.dcnn_model.score(self.valid_X,self.valid_Y))

def svm_poly_fit(self):
    # 建立 svm kernel = poly 模型
    self.poly_svc_model = svm.SVC(C=0.5, max_iter=3000,kernel='poly',probability=True)
    # 使用訓練資料訓練模型
    self.poly_svc_model.fit(self.train_X, self.train_Y)
    # 使用訓練資料預測分類
    self.poly_train_predicted = self.poly_svc_model.predict(self.train_X)
    self.poly_test_predicted = self.poly_svc_model.predict(self.valid_X)

    self.poly_train_predicted_prob = self.poly_svc_model.predict_proba(self.train_X)
    self.poly_test_predicted_prob = self.poly_svc_model.predict_proba(self.valid_X)

def svm_rbf_fit(self):
    classes = np.unique(self.train_Y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=self.train_Y)
    class_weight_dict = dict(zip(classes, weights))
    print("使用的 class_weight:", class_weight_dict)

    # 加入 class_weight 參數
    self.rbf_svc_model = svm.SVC(
        C=0.5,
        max_iter=3000,
        kernel='rbf',
        probability=True,
        class_weight='balanced'  # 或 class_weight='balanced'
    )
    
    self.rbf_svc_model.fit(self.train_X, self.train_Y)
    # 使用訓練資料預測分類
    self.rbf_train_predicted = self.rbf_svc_model.predict(self.train_X)
    self.rbf_test_predicted = self.rbf_svc_model.predict(self.valid_X)

    self.rbf_train_predicted_prob = self.rbf_svc_model.predict_proba(self.train_X)
    self.rbf_test_predicted_prob = self.rbf_svc_model.predict_proba(self.valid_X)

    # 計算準確率
    print('rbf訓練集: ',self.rbf_svc_model.score(self.train_X,self.train_Y))
    print('rbf測試集: ',self.rbf_svc_model.score(self.valid_X,self.valid_Y))
    print('========================')

def decision_tree_fit(self):
    from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score
    """
    criterion: 亂度的評估標準 gini/entropy。預設為gini。

    max_depth: 樹的最大深度。

    splitter: 特徵劃分點選擇標準 best/random。預設為best。

    random_state: 亂數種子 確保每次訓練結果都一樣 splitter=random 才有用。

    min_samples_split: 至少有多少資料才能再分

    min_samples_leaf: 分完至少有多少資料才能分
    """
    recall_scorer = make_scorer(recall_score, pos_label=1)

    param_grid = {'max_depth': [3, 5, 7, 10, None]}
    grid_search = GridSearchCV(DecisionTreeClassifier(criterion='entropy', random_state=42), param_grid, cv=5,scoring=recall_scorer)
    grid_search.fit(self.train_X, self.train_Y)

    print("最佳 max_depth:", grid_search.best_params_['max_depth'])
    self.decision_tree_model = grid_search.best_estimator_
    # 使用訓練資料訓練模型
    self.decision_tree_model.fit(self.train_X, self.train_Y)
    
    # 使用訓練資料預測分類
    self.decision_train_predicted = self.decision_tree_model.predict(self.train_X)
    self.decision_test_predicted = self.decision_tree_model.predict(self.valid_X)

    self.decision_train_predicted_prob = self.decision_tree_model.predict_proba(self.train_X)
    self.decision_test_predicted_prob = self.decision_tree_model.predict_proba(self.valid_X)

    # 🧪 使用自訂閾值分類
    threshold = 0.3  # ⬅️ 你可以調整這個值
    test_prob = self.decision_test_predicted_prob[:, 1]
    test_pred_thresh = (test_prob >= threshold).astype(int)

    # ✅ 額外指標報告
    # precision = precision_score(self.valid_Y, test_pred_thresh)
    # recall = recall_score(self.valid_Y, test_pred_thresh)
    # f1 = f1_score(self.valid_Y, test_pred_thresh)
    # for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     pred = (test_prob >= t).astype(int)
    #     f1 = f1_score(self.valid_Y, pred)
    #     print(f"Threshold={t:.2f} → F1 Score={f1:.2f}")

    # 計算準確率
    print('訓練集: ',self.decision_tree_model.score(self.train_X,self.train_Y))
    print('測試集: ',self.decision_tree_model.score(self.valid_X,self.valid_Y))

def random_forest_fit(self):
    """
    n_estimators: 森林中樹木的數量，預設=100。

    max_features: 劃分時考慮的最大特徵數 預設auto。

    criterion: 亂度的評估標準 gini/entropy。預設為gini。

    max_depth: 樹的最大深度。

    splitter: 特徵劃分點選擇標準 best/random。預設為best。

    random_state: 亂數種子 確保每次訓練結果都一樣 splitter=random 才有用。

    min_samples_split: 至少有多少資料才能再分

    min_samples_leaf: 分完至少有多少資料才能分
    """
    from imblearn.ensemble import BalancedRandomForestClassifier

    # 建立 Balanced Random Forest 模型
    self.forest_model = BalancedRandomForestClassifier(
        criterion='entropy',
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        sampling_strategy='all',   # 預設 equal sampling，建議保留
        replacement=True,          # 和 bootstrap 對應使用
        bootstrap=False,           # 0.13 預設改為 False
        random_state=42            # 可設為固定值以重現結果
    )

    # 訓練模型
    self.forest_model.fit(self.train_X, self.train_Y)

    # 預測
    self.forest_train_predicted = self.forest_model.predict(self.train_X)
    self.forest_test_predicted = self.forest_model.predict(self.valid_X)

    self.forest_train_predicted_prob = self.forest_model.predict_proba(self.train_X)
    self.forest_test_predicted_prob = self.forest_model.predict_proba(self.valid_X)

    # 預測成功的比例 (accuracy)
    print('訓練集: ', self.forest_model.score(self.train_X, self.train_Y))
    print('測試集: ', self.forest_model.score(self.valid_X, self.valid_Y))


def xgboost_fit(self):
    # 建立 XGBClassifier 模型
    self.xgboost_model = XGBClassifier(n_estimators=300,
                                        booster='gbtree',
                                        learning_rate=0.1,
                                        max_depth=7,
                                        min_child_weight=5,
                                        gamma=0.0,
                                        subsample=0.9,
                                        colsample_bytree=0.8,
                                        objective='reg:logistic',
                                        tree_method='hist',
                                        reg_alpha=0.0,
                                        reg_lambda=1.0,
                                        eval_metric='auc',
                                        nthread=2,
                                        random_state=0,
                                        scale_pos_weight=1,
                                        seed=0)
    # 使用訓練資料訓練模型
    self.xgboost_model.fit(self.train_X, self.train_Y)
    # 使用訓練資料預測分類
    self.xgboost_train_predicted = self.xgboost_model.predict(self.train_X)
    self.xgboost_test_predicted = self.xgboost_model.predict(self.valid_X)

    self.xgboost_train_predicted_prob = self.xgboost_model.predict_proba(self.train_X)
    self.xgboost_test_predicted_prob = self.xgboost_model.predict_proba(self.valid_X)

    print('訓練集: ',self.xgboost_model.score(self.train_X,self.train_Y))
    print('測試集: ',self.xgboost_model.score(self.valid_X,self.valid_Y))

def adaboost_fit(self):
    self.adaboost_model = AdaBoostClassifier(n_estimators = 300)
    self.adaboost_model.fit(self.train_X,self.train_Y)

    self.adaboost_train_predicted = self.adaboost_model.predict(self.train_X)
    self.adaboost_test_predicted = self.adaboost_model.predict(self.valid_X)

    self.adaboost_ada_train_predicted_prob = self.adaboost_model.predict_proba(self.train_X)
    self.adaboost_test_predicted_prob = self.adaboost_model.predict_proba(self.valid_X)

    print("訓練集 Score: ", self.adaboost_model.score(self.train_X,self.train_Y))
    print("測試集 Score: ", self.adaboost_model.score(self.valid_X,self.valid_Y))

def gradient_boost_fit(self):
    # Initialize and train GradientBoostingClassifier
    self.grad_boost_model = GradientBoostingClassifier(n_estimators=100, random_state=0)
    self.grad_boost_model.fit(self.train_X, self.train_Y)

    # Make predictions
    self.grad_boost_train_predicted = self.grad_boost_model.predict(self.train_X)
    self.grad_boost_test_predicted = self.grad_boost_model.predict(self.valid_X)

    # predict_proba for class probabilities
    self.grad_boost_train_predicted_prob = self.grad_boost_model.predict_proba(self.train_X)
    self.grad_boost_test_predicted_prob = self.grad_boost_model.predict_proba(self.valid_X)

    # Print scores
    print("Training Set Score:", accuracy_score(self.train_Y, self.grad_boost_train_predicted))
    print("Testing Set Score:", accuracy_score(self.valid_Y, self.grad_boost_test_predicted))
