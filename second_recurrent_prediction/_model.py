import numpy as np
import pandas as pd
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
    self.linear_svc_model = svm.SVC(C=0.5, max_iter=3000,kernel='linear',probability=True)
    # 使用訓練資料訓練模型
    self.linear_svc_model.fit(self.train_X, self.train_Y)
    # 使用訓練資料預測分類
    self.linear_train_predicted=self.linear_svc_model.predict(self.train_X)
    self.linear_test_predicted=self.linear_svc_model.predict(self.test_X)

    # 使用訓練資料預測機率S
    self.linear_train_predicted_prob =self.linear_svc_model.predict_proba(self.train_X)
    self.linear_test_predicted_prob =self.linear_svc_model.predict_proba(self.test_X)

    # 計算準確率
    print('linear訓練集: ',self.linear_svc_model.score(self.train_X,self.train_Y))
    print('linear測試集: ',self.linear_svc_model.score(self.test_X,self.test_Y))
    print('========================')

def svm_poly_fit(self):
    # 建立 svm kernel = poly 模型
    self.poly_svc_model = svm.SVC(C=0.5, max_iter=3000,kernel='poly',probability=True)
    # 使用訓練資料訓練模型
    self.poly_svc_model.fit(self.train_X, self.train_Y)
    # 使用訓練資料預測分類
    self.poly_train_predicted = self.poly_svc_model.predict(self.train_X)
    self.poly_test_predicted = self.poly_svc_model.predict(self.test_X)

    self.poly_train_predicted_prob = self.poly_svc_model.predict_proba(self.train_X)
    self.poly_test_predicted_prob = self.poly_svc_model.predict_proba(self.test_X)

def svm_rbf_fit(self):
    self.rbf_svc_model = svm.SVC(C=0.5, max_iter=3000,kernel='rbf',probability=True)
    # 使用訓練資料訓練模型
    self.rbf_svc_model.fit(self.train_X, self.train_Y)
    # 使用訓練資料預測分類
    self.rbf_train_predicted = self.rbf_svc_model.predict(self.train_X)
    self.rbf_test_predicted = self.rbf_svc_model.predict(self.test_X)

    self.rbf_train_predicted_prob = self.rbf_svc_model.predict_proba(self.train_X)
    self.rbf_test_predicted_prob = self.rbf_svc_model.predict_proba(self.test_X)

    # 計算準確率
    print('rbf訓練集: ',self.rbf_svc_model.score(self.train_X,self.train_Y))
    print('rbf測試集: ',self.rbf_svc_model.score(self.test_X,self.test_Y))
    print('========================')

def decision_tree_fit(self):
    """
    criterion: 亂度的評估標準 gini/entropy。預設為gini。

    max_depth: 樹的最大深度。

    splitter: 特徵劃分點選擇標準 best/random。預設為best。

    random_state: 亂數種子 確保每次訓練結果都一樣 splitter=random 才有用。

    min_samples_split: 至少有多少資料才能再分

    min_samples_leaf: 分完至少有多少資料才能分
    """
    param_grid = {'max_depth': [3, 5, 7, 10, None]}
    grid_search = GridSearchCV(DecisionTreeClassifier(criterion='entropy', random_state=42), param_grid, cv=5)
    grid_search.fit(self.train_X, self.train_Y)

    print("最佳 max_depth:", grid_search.best_params_['max_depth'])
    self.decision_tree_model = grid_search.best_estimator_
    # 使用訓練資料訓練模型
    self.decision_tree_model.fit(self.train_X, self.train_Y)
    # 使用訓練資料預測分類
    self.decision_train_predicted = self.decision_tree_model.predict(self.train_X)
    self.decision_test_predicted = self.decision_tree_model.predict(self.test_X)

    self.decision_train_predicted_prob = self.decision_tree_model.predict_proba(self.train_X)
    self.decision_test_predicted_prob = self.decision_tree_model.predict_proba(self.test_X)

    # 計算準確率
    print('訓練集: ',self.decision_tree_model.score(self.train_X,self.train_Y))
    print('測試集: ',self.decision_tree_model.score(self.test_X,self.test_Y))

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
    #建立隨機森林 model
    self.forest_model = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, max_depth=5)
    forest_fit = self.forest_model.fit(self.train_X, self.train_Y)

    # 預測
    self.forest_train_predicted = self.forest_model.predict(self.train_X)
    self.forest_test_predicted = self.forest_model.predict(self.test_X)

    self.forest_train_predicted_prob = self.forest_model.predict_proba(self.train_X)
    self.forest_test_predicted_prob = self.forest_model.predict_proba(self.test_X)

    # 預測成功的比例
    print('訓練集: ',self.forest_model.score(self.train_X,self.train_Y))
    print('測試集: ',self.forest_model.score(self.test_X,self.test_Y))

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
    self.xgboost_test_predicted = self.xgboost_model.predict(self.test_X)

    self.xgboost_train_predicted_prob = self.xgboost_model.predict_proba(self.train_X)
    self.xgboost_test_predicted_prob = self.xgboost_model.predict_proba(self.test_X)

    print('訓練集: ',self.xgboost_model.score(self.train_X,self.train_Y))
    print('測試集: ',self.xgboost_model.score(self.test_X,self.test_Y))

def adaboost_fit(self):
    self.adaboost_model = AdaBoostClassifier(n_estimators = 300)
    self.adaboost_model.fit(self.train_X,self.train_Y)

    self.adaboost_train_predicted = self.adaboost_model.predict(self.train_X)
    self.adaboost_test_predicted = self.adaboost_model.predict(self.test_X)

    self.adaboost_ada_train_predicted_prob = self.adaboost_model.predict_proba(self.train_X)
    self.adaboost_test_predicted_prob = self.adaboost_model.predict_proba(self.test_X)

    print("訓練集 Score: ", self.adaboost_model.score(self.train_X,self.train_Y))
    print("測試集 Score: ", self.adaboost_model.score(self.test_X,self.test_Y))

def gradient_boost_fit(self):
    # Initialize and train GradientBoostingClassifier
    self.grad_boost_model = GradientBoostingClassifier(n_estimators=100, random_state=0)
    self.grad_boost_model.fit(self.train_X, self.train_Y)

    # Make predictions
    self.grad_boost_train_predicted = self.grad_boost_model.predict(self.train_X)
    self.grad_boost_test_predicted = self.grad_boost_model.predict(self.test_X)

    # predict_proba for class probabilities
    self.grad_boost_train_predicted_prob = self.grad_boost_model.predict_proba(self.train_X)
    self.grad_boost_test_predicted_prob = self.grad_boost_model.predict_proba(self.test_X)

    # Print scores
    print("Training Set Score:", accuracy_score(self.train_Y, self.grad_boost_train_predicted))
    print("Testing Set Score:", accuracy_score(self.test_Y, self.grad_boost_test_predicted))
