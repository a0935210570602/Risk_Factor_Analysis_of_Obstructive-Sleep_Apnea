import itertools
import gc
import warnings
import multiprocessing
from exp_config import (smote_methods, feature_selection_methods,
                        prediction_methods, file_paths)
from second_recurrent_prediction import SecondStrokePrediction
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning, UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The least populated class in y has only")

def single_experiment(args):
    """每次在子進程要執行的程式。"""
    (file_path, smote_method,
     feature_selection_method, prediction_method) = args

    print(f"🚀 Running: {file_path}, smote={smote_method}, "
          f"fs={feature_selection_method.__name__}, pred={prediction_method.__name__}")

    try:
        model_prediction = SecondStrokePrediction(file_path)
        model_prediction.load_data()
        model_prediction.prepare_tenfold_data()
        model_prediction.apply_standardization()
        model_prediction.set_smote_method(smote_method)

        # smote_method(model_prediction)
        # feature_selection_method(model_prediction)
        model_prediction.set_prediction_model(prediction_method)
        model_prediction.cross_validation()
    finally:
        model_prediction.clear_data_and_model()
        gc.collect()  # 每次子進程結束時釋放資源

def run_experiments():
    all_params = []
    for file_path in file_paths:
        for combo in itertools.product(
            # down_sampling_rates,
            smote_methods,
            feature_selection_methods,
            prediction_methods
        ):
            smote_method, feature_selection_method, prediction_method = combo
            all_params.append((file_path,
                            #    down_sampling_rate,
                               smote_method,
                               feature_selection_method,
                               prediction_method))

    for i, param in enumerate(all_params):
        print(f"🔥 Executing experiment {i+1}/{len(all_params)}")
        p = multiprocessing.Process(target=single_experiment, args=(param,))
        p.start()
        p.join()  # 等待子進程執行完畢再繼續，保證執行完即關閉
        print(f"✅ Finished experiment {i+1}")

if __name__ == '__main__':
    run_experiments()
