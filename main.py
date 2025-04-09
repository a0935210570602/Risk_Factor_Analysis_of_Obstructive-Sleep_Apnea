import itertools
import gc
import warnings
from concurrent.futures import ProcessPoolExecutor

from exp_config import (smote_methods, feature_selection_methods,
                        prediction_methods, down_sampling_rates, file_paths)
from second_recurrent_prediction import SecondStrokePrediction
# 關閉一些不必要的警告
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The least populated class in y has only")

# 單次實驗函式，放在模組層級以利子進程呼叫
def single_experiment(args):
    """每次在子進程要執行的程式。"""
    (file_path, down_sampling_rate, smote_method,
     feature_selection_method, prediction_method) = args

    # 這裡可以再包一層 try...finally, 如果需要確保釋放資源
    model_prediction = SecondStrokePrediction(file_path)
    model_prediction.set_downsampling_rate(down_sampling_rate)
    model_prediction.load_data()
    model_prediction.standardize_data()

    smote_method(model_prediction)
    feature_selection_method(model_prediction)
    prediction_method(model_prediction)

    del model_prediction
    gc.collect()  # 垃圾回收（即使不一定釋放給 OS，但子進程結束會徹底釋放）

def run_experiments():
    # 建立所有參數組合
    all_params = []
    for file_path in file_paths:
        for combo in itertools.product(
            down_sampling_rates,
            smote_methods,
            feature_selection_methods,
            prediction_methods
        ):
            down_sampling_rate, smote_method, feature_selection_method, prediction_method = combo
            all_params.append((file_path,
                               down_sampling_rate,
                               smote_method,
                               feature_selection_method,
                               prediction_method))

    # 使用多進程池，一個參數組合對應一個子進程
    with ProcessPoolExecutor(max_workers=1) as executor:
        # max_workers=1 表示一次只執行一個 child process，避免佔用過多記憶體
        # 如有足夠資源，可調整到 2、4 或更多
        list(executor.map(single_experiment, all_params))

if __name__ == '__main__':
    run_experiments()
