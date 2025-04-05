import pandas as pd
from .model_config_parser import ModelConfigParser
from .model_builder import ModelBuilder
from .data_processor import DataProcessor
from lib import balance_data

class ExperimentPipeline:
    def __init__(self, exp_config):
        self.exp_config = exp_config

    def data_process(self, data_config):
        processor = DataProcessor(data_config)
        self.train_X, self.train_Y, self.test_X, self.test_Y = processor.load_data()
        # 檢查是否需要進行資料平衡處理
        if data_config.get("balance") == "smote":
            self.train_X, self.train_Y = balance_data.smote(self.train_X, self.train_Y)
    
    def model_process(self):
        model_results = []
        for config in self.parsed_configs:
            runs = config["runs"]
            for run in range(runs):
                # 根據配置建立模型實例
                model_instance = ModelBuilder.build(config)
                model_instance.train(self.train_X, self.train_Y)
                model_instance.predict(self.train_X)
                train_evaluation = model_instance.evaluate(self.train_Y)
                model_instance.save_result()

                model_results.append({
                    "model": config["model_name"],
                    "run": run + 1,
                    "acc": train_evaluation.get("acc"),
                    "recall": train_evaluation.get("recall"),
                    "precision": train_evaluation.get("precision"),
                    "auc": train_evaluation.get("auc"),
                    "f1": train_evaluation.get("f1"),
                    "model_config": str(config.get("params", {})),
                    "state": "train"
                })

                model_instance.predict(self.test_X)
                test_evaluation = model_instance.evaluate(self.test_Y)
                model_instance.save_result()

                model_results.append({
                    "model": config["model_name"],
                    "run": run + 1,
                    "acc": test_evaluation.get("acc"),
                    "recall": test_evaluation.get("recall"),
                    "precision": test_evaluation.get("precision"),
                    "auc": test_evaluation.get("auc"),
                    "f1": test_evaluation.get("f1"),
                    "model_config": str(config.get("params", {})),
                    "state": "test"
                })

        return model_results

    def run(self):
        all_results = []
        exp_num = 1

        for exp in self.exp_config:
            data_config = exp.get("data_config", {})
            model_configs = exp.get("model_config", {})
            parser = ModelConfigParser(model_configs)
            self.parsed_configs = parser.parse()
            # 資料處理：讀取並拆分數據
            self.data_process(data_config)
            # 模型處理：建立、訓練、評估模型
            exp_results = self.model_process()
            # 為每個結果添加實驗編號、資料路徑與平衡策略信息
            for res in exp_results:
                res["experiment"] = f"exp{exp_num}"
                res["data_path"] = data_config.get("path")
                res["balance"] = data_config.get("balance", None)
                all_results.append(res)
            exp_num += 1
                
        # 儲存所有結果到 CSV 檔
        df = pd.DataFrame(all_results)
        # 指定想要的欄位順序，將 'experiment' 放在第一欄
        desired_columns = [
            "experiment",
            "balance",
            "model",
            "state",
            "model_config",
            "run",
            "acc",
            "recall",
            "precision",
            "auc",
            "f1",
            "data_path",
        ]
        # 重新排列欄位
        df = df[desired_columns]
        # 輸出 CSV
        df.to_csv("experiment_results.csv", index=False)
        
        return all_results
