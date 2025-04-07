import pandas as pd

def predict_svm_linear(self):
    self.load_data()
    self.svm_linear_fit()
    name = 'SVM Linear'
    train_title = f'{name} Train'
    test_title = f'{name} Test'
    self.confusion_matrix(self.train_Y, self.linear_train_predicted, train_title)
    self.confusion_matrix(self.test_Y, self.linear_test_predicted, test_title)
    self.svm_linear_train_auc,self.svm_linear_train_fpr, self.svm_linear_train_tpr = self.plot_ROC_curve(self.train_Y, self.linear_train_predicted_prob[:,1], train_title)
    self.svm_linear_test_auc,self.svm_linear_test_fpr, self.svm_linear_test_tpr = self.plot_ROC_curve(self.test_Y, self.linear_test_predicted_prob[:,1], test_title)
    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.svm_linear_train_auc, 
         'fpr': self.svm_linear_train_fpr, 'tpr': self.svm_linear_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.svm_linear_test_auc, 
         'fpr': self.svm_linear_test_fpr, 'tpr': self.svm_linear_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(self.linear_train_predicted, name, self.svm_linear_train_auc)
    self.gen_result(self.linear_test_predicted, name, self.svm_linear_test_auc, is_train=False)

    self.cross_validation(self.linear_svc_model,name)

def predict_svm_poly(self):
    self.load_data()
    self.svm_poly_fit()
    name = 'SVM Poly'
    train_title = f'{name} Train'
    test_title = f'{name} Test'
    self.confusion_matrix(self.train_Y, self.poly_train_predicted, train_title)
    self.confusion_matrix(self.test_Y, self.poly_test_predicted, test_title)
    self.svm_poly_train_auc, self.svm_poly_train_fpr, self.svm_poly_train_tpr = self.plot_ROC_curve(self.train_Y, self.poly_train_predicted_prob[:,1].reshape(-1,1), train_title)
    self.svm_poly_test_auc, self.svm_poly_test_fpr, self.svm_poly_test_tpr = self.plot_ROC_curve(self.test_Y, self.poly_test_predicted_prob[:,1].reshape(-1,1), test_title)
    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.svm_poly_train_auc, 
         'fpr': self.svm_poly_train_fpr, 'tpr': self.svm_poly_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.svm_poly_test_auc, 
         'fpr': self.svm_poly_test_fpr, 'tpr': self.svm_poly_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(self.poly_train_predicted ,name, self.svm_poly_train_auc)
    self.gen_result(self.poly_test_predicted ,name, self.svm_poly_test_auc, is_train=False)
    self.cross_validation(self.poly_svc_model,name)

def predict_svm_rbf(self):
    self.load_data()
    self.svm_rbf_fit()
    name = 'SVM RBF'
    train_title = f'{name} Trian'
    test_title = f'{name} Test'
    self.confusion_matrix(self.train_Y,self.rbf_train_predicted,train_title)
    self.confusion_matrix(self.test_Y,self.rbf_test_predicted,test_title)
    self.svm_rbf_train_auc,self.svm_rbf_train_fpr, self.svm_rbf_train_tpr = self.plot_ROC_curve(self.train_Y,self.rbf_train_predicted_prob[:,1],train_title)
    self.svm_rbf_test_auc,self.svm_rbf_test_fpr, self.svm_rbf_test_tpr = self.plot_ROC_curve(self.test_Y,self.rbf_test_predicted_prob[:,1],test_title)
    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.svm_rbf_train_auc, 
         'fpr': self.svm_rbf_train_fpr, 'tpr': self.svm_rbf_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.svm_rbf_test_auc, 
         'fpr': self.svm_rbf_test_fpr, 'tpr': self.svm_rbf_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(self.rbf_train_predicted ,name,self.svm_rbf_train_auc)
    self.gen_result(self.rbf_test_predicted ,name,self.svm_rbf_test_auc, is_train=False)
    self.cross_validation(self.rbf_svc_model,name)

def predict_decision_tree(self):
    self.load_data()
    self.decision_tree_fit()
    name = 'Decision Tree'
    train_title = f'{name} Trian'
    test_title = f'{name} Test'
    self.confusion_matrix(self.train_Y,self.decision_train_predicted,train_title)
    self.confusion_matrix(self.test_Y,self.decision_test_predicted,test_title)
    self.decision_tree_train_auc, self.decision_tree_train_fpr, self.decision_tree_train_tpr = self.plot_ROC_curve(self.train_Y,self.decision_train_predicted_prob[:,1],train_title)
    self.decision_tree_test_auc, self.decision_tree_test_fpr, self.decision_tree_test_tpr = self.plot_ROC_curve(self.test_Y,self.decision_test_predicted_prob[:,1],test_title)
    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.decision_tree_train_auc, 
         'fpr': self.decision_tree_train_fpr, 'tpr': self.decision_tree_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.decision_tree_test_auc, 
         'fpr': self.decision_tree_test_fpr, 'tpr': self.decision_tree_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(self.decision_train_predicted ,name,self.decision_tree_train_auc)
    self.gen_result(self.decision_test_predicted ,name,self.decision_tree_test_auc, is_train=False)
    self.cross_validation(self.decision_tree_model,name)

    importance_list = self.decision_tree_model.feature_importances_
    self.plot_feature_importance_bar_chart(importance_list, 'decision_tree', name)
    self.plot_tree_graph(is_forest=False)

def predict_random_forest(self):
    self.load_data()
    self.random_forest_fit()
    name = 'Random Forest'
    train_title = f'{name} Trian'
    test_title = f'{name} Test'
    self.confusion_matrix(self.train_Y,self.forest_train_predicted,train_title)
    self.confusion_matrix(self.test_Y,self.forest_test_predicted,test_title)
    self.random_forest_train_auc, self.random_forest_train_fpr, self.random_forest_train_tpr = self.plot_ROC_curve(self.train_Y,self.forest_train_predicted_prob[:,1],train_title)
    self.random_forest_test_auc, self.random_forest_test_fpr, self.random_forest_test_tpr = self.plot_ROC_curve(self.test_Y,self.forest_test_predicted_prob[:,1],test_title)
    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.random_forest_train_auc, 
         'fpr': self.random_forest_train_fpr, 'tpr': self.random_forest_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.random_forest_test_auc, 
         'fpr': self.random_forest_test_fpr, 'tpr': self.random_forest_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(self.forest_train_predicted, name, self.random_forest_train_auc)
    self.gen_result(self.forest_test_predicted, name, self.random_forest_test_auc, is_train=False)
    self.cross_validation(self.forest_model,name)

    importance_list = self.forest_model.feature_importances_
    self.plot_feature_importance_bar_chart(importance_list, 'random_forest', name)
    self.plot_tree_graph(is_forest=True)

def predict_xgboost(self):
    self.load_data()
    self.xgboost_fit()
    name = 'XGBoost'
    train_title = f'{name} Trian'
    test_title = f'{name} Test'
    self.confusion_matrix(self.train_Y,self.xgboost_train_predicted,train_title)
    self.confusion_matrix(self.test_Y,self.xgboost_test_predicted,test_title)
    self.xgboost_train_auc, self.xgboost_train_fpr, self.xgboost_train_tpr = self.plot_ROC_curve(self.train_Y,self.xgboost_train_predicted_prob[:,1],train_title)
    self.xgboost_test_auc, self.xgboost_test_fpr, self.xgboost_test_tpr = self.plot_ROC_curve(self.test_Y,self.xgboost_test_predicted_prob[:,1],test_title)
    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.xgboost_train_auc, 
         'fpr': self.xgboost_train_fpr, 'tpr': self.xgboost_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.xgboost_test_auc, 
         'fpr': self.xgboost_test_fpr, 'tpr': self.xgboost_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(self.xgboost_train_predicted, name, self.xgboost_train_auc)
    self.gen_result(self.xgboost_test_predicted ,name, self.xgboost_test_auc, is_train=False)
    self.cross_validation(self.xgboost_model,name)

    self.plot_xgboost_feature_importance()

def predict_adaboost(self):
    self.load_data()
    self.adaboost_fit()
    name = 'AdaBoost'
    train_title = f'{name} Trian'
    test_title = f'{name} Test'
    self.confusion_matrix(self.train_Y,self.adaboost_train_predicted,train_title)
    self.confusion_matrix(self.test_Y,self.adaboost_test_predicted,test_title)
    self.adaboost_train_auc, self.adaboost_train_fpr, self.adaboost_train_tpr = self.plot_ROC_curve(self.train_Y,self.adaboost_ada_train_predicted_prob[:,1],train_title)
    self.adaboost_test_auc, self.adaboost_test_fpr, self.adaboost_test_tpr = self.plot_ROC_curve(self.test_Y,self.adaboost_test_predicted_prob[:,1],test_title)
    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.adaboost_train_auc, 
         'fpr': self.adaboost_train_fpr, 'tpr': self.adaboost_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.adaboost_test_auc, 
         'fpr': self.adaboost_test_fpr, 'tpr': self.adaboost_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(self.adaboost_train_predicted, name, self.adaboost_train_auc)
    self.gen_result(self.adaboost_test_predicted, name, self.adaboost_test_auc, is_train=False)
    self.cross_validation(self.adaboost_model, name)

    importance_list = self.adaboost_model.feature_importances_
    self.plot_feature_importance_bar_chart(importance_list, 'adaboost', name)

def predict_gradient_boost(self):
    self.load_data()
    self.gradient_boost_fit()
    name = 'Gradient Boost'
    train_title = f'{name} Trian'
    test_title = f'{name} Test'
    self.confusion_matrix(self.train_Y,self.grad_boost_train_predicted,train_title)
    self.confusion_matrix(self.test_Y,self.grad_boost_test_predicted,test_title)
    self.grad_boost_train_auc, self.grad_boost_train_fpr, self.grad_boost_train_tpr = self.plot_ROC_curve(self.train_Y,self.grad_boost_train_predicted_prob[:,1],train_title)
    self.grad_boost_test_auc, self.grad_boost_test_fpr, self.grad_boost_test_tpr = self.plot_ROC_curve(self.test_Y,self.grad_boost_test_predicted_prob[:,1],test_title)
    roc_curve_result_df = pd.DataFrame([
        {'model': name, 'dataset': 'train', 'auc': self.grad_boost_train_auc, 
         'fpr': self.grad_boost_train_fpr, 'tpr': self.grad_boost_train_tpr},
        {'model': name, 'dataset': 'test', 'auc': self.grad_boost_test_auc, 
         'fpr': self.grad_boost_test_fpr, 'tpr': self.grad_boost_test_tpr},
    ])
    self.roc_curve_result_df = pd.concat([self.roc_curve_result_df, roc_curve_result_df], ignore_index=True)

    self.gen_result(self.grad_boost_train_predicted, name, self.grad_boost_train_auc)
    self.gen_result(self.grad_boost_test_predicted, name, self.grad_boost_test_auc, is_train=False)
    self.cross_validation(self.grad_boost_model,name)

    importance_list = self.grad_boost_model.feature_importances_
    self.plot_feature_importance_bar_chart(importance_list, 'gradien_boost', name)

def show_all_result(self):
    self.display_total_result_train()
    self.display_total_result_test()
    self.display_total_ten_fold_result()
    self.plot_total_ROC_curve(is_train=True)
    self.plot_total_ROC_curve(is_train=False)
