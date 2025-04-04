import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

import csv
from sklearn.model_selection import train_test_split
from sklearn import  ensemble, preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
import os
from sklearn.model_selection import cross_validate

def confusion_matrix(self, true_y, pred_y, name):
    dir_path = os.path.join(self.PATH, 'confusion_matrix_image')
    file_path = os.path.join(dir_path, f'{name}.png')
    if not os.path.isdir(dir_path):  # 確認儲存檔案位置 若沒有的話 則新建檔案
        os.makedirs(dir_path)

    cm = metrics.confusion_matrix(true_y, pred_y) #y_true為data真實值  model為預測結果
    plt.matshow(cm, cmap=plt.cm.BuGn) #畫圖
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[i,j], xy=(i, j), horizontalalignment='center', verticalalignment='center') # 文字註解
    plt.title(name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(file_path, bbox_inches='tight') #存檔
    plt.show()

def plot_ROC_curve(self, true_y, pred_y, name):
    dir_path = os.path.join(self.PATH,'ROC_curve_image')
    file_path = os.path.join(dir_path, f'{name}.png')
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    fpr, tpr, thersholds = roc_curve(true_y,pred_y)

    roc_auc = auc(fpr,tpr)
    plt.plot(fpr, tpr,color='darkorange', label= 'ROC(area = {0:.2f})'.format(roc_auc),lw=2)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Ture Positive Rate')
    plt.title(name+' ROC Curve')
    plt.legend(loc = 'lower right')
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()

    return roc_auc,fpr,tpr

def plot_feature_importance_bar_chart(self, importances, file_name, img_title):
    # 获取特征名称
    feature_names = self.train_X.columns
    # 将特征重要性进行排序
    indices = np.argsort(importances)[::-1]

    dir_path = os.path.join(self.PATH, 'feature_importances_image')
    file_path = os.path.join(dir_path, f'{file_name}.png')
    if not os.path.isdir(dir_path):  # 確認儲存檔案位置 若沒有的話 則新建檔案
        os.makedirs(dir_path)
    file_path = os.path.join(self.PATH, file_path)

    # 画出特征重要性图表
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(self.train_X.shape[1]), importances[indices], align="center")
    plt.xticks(range(self.train_X.shape[1]), feature_names[indices], rotation=45)
    plt.title(f"Feature Importance - {img_title}")
    plt.xlabel("Feature")
    plt.ylabel("Importance")

    # 在每个条形上标示特征重要值数字
    for bar, importance in zip(bars, importances[indices]):
        yval = round(importance, 3)
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, yval, ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight') #存檔
    plt.show()

def plot_tree_graph(self, is_forest = False):
    # self.decision_tree_model.classes_ = ['0','1']
    # tree.export_graphviz(self.decision_tree_model, out_file=self.PATH+'decisiontree.dot',
    #                 feature_names=self.SELECTTED_FEATURE_LIST,
    #                 class_names=self.decision_tree_model.classes_)
    idx = 5 if is_forest else 1
    model = self.forest_model if is_forest else self.decision_tree_model
    for i in range(0,idx):
        fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (2,2), dpi=1000)
        estimator = model.estimators_[i] if is_forest else model
        tree.plot_tree(estimator,
                feature_names = self.SELECTED_FEATURE_LIST,
                class_names=['No','Yes'],
                filled = True)
        img_name = f'random_tree_{i+1}.png' if is_forest else f'decision_tree.png'
        fig.savefig(os.path.join(self.PATH, img_name))

def plot_xgboost_feature_importance(self):
    dir_path = os.path.join(self.PATH, 'feature_importances_image')
    file_path = os.path.join(dir_path, f'xgboost.png')
    if not os.path.isdir(dir_path):  # 確認儲存檔案位置 若沒有的話 則新建檔案
        os.makedirs(dir_path)
    file_path = os.path.join(self.PATH, file_path)

    # 使用plot_importance函數顯示特徵重要性圖表
    plt.figure(figsize=(10, 6))
    ax = xgb.plot_importance(self.xgboost_model, importance_type='weight', show_values=True, ax=plt.gca())
    plt.title("Feature Importance - XGBoost")
    plt.xlabel("Feature importance score")
    plt.ylabel("Input variable")
    plt.grid(False)

    # 調整圖表邊框大小
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1)  # 設置邊框寬度為2個單位

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight') #存檔
    plt.show()

def plot_total_ROC_curve(self, is_train=True):
    dataset = 'train' if is_train else 'test'
    title = 'Train' if is_train else 'Test'
    result_df = self.roc_curve_result_df[self.roc_curve_result_df['dataset'] == dataset]
    plt.figure(1)
    for row in result_df.itertuples(index=True):
        plt.plot(row.fpr, row.tpr, label= '{} (AUC = {:.2f})'.format(row.model, row.auc),lw=2)

    # plt.plot(self.svm_, tpr_train3, label= 'SVM linear(AUC = {0:.2f})'.format(train_auc3),lw=2)
    # plt.plot(fpr_train4, tpr_train4, label= 'SVM poly(AUC = {0:.2f})'.format(train_auc4),lw=2)
    # plt.plot(fpr_train5, tpr_train5, label= 'SVM rbf(AUC = {0:.2f})'.format(train_auc5),lw=2)

    # plt.plot(fpr_train7, tpr_train7, label= 'Decision Tree(AUC = {0:.2f})'.format(train_auc7),lw=2)
    # plt.plot(fpr_train8, tpr_train8, label= 'Random Forest(AUC = {0:.2f})'.format(train_auc8),lw=2)
    # plt.plot(fpr_train9, tpr_train9, label= 'XGBoost(AUC = {0:.2f})'.format(train_auc9),lw=2)
    # plt.plot(fpr_train10, tpr_train10, label= 'AdaBoost(AUC = {0:.2f})'.format(train_auc10),lw=2)
    #plt.plot(fpr_train11, tpr_train11, label= 'Stacking(AUC = {0:.2f})'.format(train_auc11),lw=2)
    #plt.plot(fpr_train12, tpr_train12, label= 'LGBM(AUC = {0:.2f})'.format(train_auc12),lw=2)

    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Ture Positive Rate')
    plt.title(f'{title} ROC Curve')
    plt.legend(loc = 'lower right')
    file_path = os.path.join(self.PATH, 'ROC_curve_image', f'roc_total_{dataset}.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()