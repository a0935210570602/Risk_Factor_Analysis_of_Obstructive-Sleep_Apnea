from second_recurrent_prediction import SecondStrokePrediction

file_paths = [
    'raw_data/age_below_66.csv',
    # 'raw_data/age_between_66_73_samples.csv',
    # 'raw_data/age_over_73.csv',

    # 'raw_data/female_data.csv',
    # 'raw_data/male_data.csv',
]
# 設定 SMOTE 方法，這邊假設方法名稱與類別內定義一致
smote_methods = [
    'smote_borderline',
    # 'smote_standard',
    # 'smote_svm',
    # 'smote_adasyn',
    # 'smote_smotenc',
    
    # SecondStrokePrediction.smote_borderline,
    # SecondStrokePrediction.smote_standard,
    # SecondStrokePrediction.smote_svm,
    # SecondStrokePrediction.smote_adasyn,
    # SecondStrokePrediction.smote_smotenc,
]

# 設定特徵選擇方法，同樣假設方法名稱與類別內定義一致
feature_selection_methods = [
    # SecondStrokePrediction.sfs_knn_feature_selection,
    # SecondStrokePrediction.sfs_linear_feature_selection,
    SecondStrokePrediction.sfs_random_forest_feature_selection,
    # SecondStrokePrediction.sfs_xgboost_feature_selection,
    # SecondStrokePrediction.permutation_feature_selection,
    # SecondStrokePrediction.boruta_feature_selection,
    # SecondStrokePrediction.rfecv_feature_selection,
    # SecondStrokePrediction.variance_threshold_selection,
    # SecondStrokePrediction.l1_feature_selection,
]

# 設定各預測模型方法
prediction_methods = [
    SecondStrokePrediction.predict_svm_linear,
    # SecondStrokePrediction.predict_svm_poly,
    # SecondStrokePrediction.predict_svm_rbf,
    # SecondStrokePrediction.predict_decision_tree,
    # SecondStrokePrediction.predict_random_forest,
    # SecondStrokePrediction.predict_dcnn,
    # SecondStrokePrediction.predict_adaboost,
    # SecondStrokePrediction.predict_gradient_boost,
    # SecondStrokePrediction.predict_xgboost,
]

# down_sampling_rates = [1]