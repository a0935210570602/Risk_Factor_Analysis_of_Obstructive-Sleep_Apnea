import pandas as pd
from model_config_parser import ModelConfigParser
from model_builder import ModelBuilder
from data_processor import DataProcessor

class ExperimentPipeline:
    def __init__(self, exp_config):
        self.exp_config = exp_config

    def data_process(self, data_config):
        processor = DataProcessor(data_config)
        self.train_X, self.train_Y, self.test_X, self.test_Y = processor.load_data()

    def model_process(self):
        model_results = []
        for config in self.parsed_configs:
            runs = config["runs"]
            for run in range(runs):
                # Build model instance according to configuration
                model_instance = ModelBuilder.build(config)
                model_instance.train(self.train_X, self.train_Y)
                model_instance.predict(self.train_X)
                evaluation = model_instance.evaluate(self.train_Y)
                model_instance.save_result()

                model_results.append({
                    "model": config["model_name"],
                    "run": run + 1,
                    "acc": evaluation.get("acc"),
                    "recall": evaluation.get("recall"),
                    "precision": evaluation.get("precision"),
                    "auc": evaluation.get("auc"),
                    "f1": evaluation.get("f1"),
                    "model_config": str(config.get("params", {}))
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
            # Data processing: load and split data
            self.data_process(data_config)
            # Model processing: build, train, evaluate models
            exp_results = self.model_process()
            # 為每個結果添加實驗編號與數據路徑信息
            for res in exp_results:
                res["experiment"] = f"exp{exp_num}"
                res["data_path"] = data_config.get("path")
                all_results.append(res)
            exp_num += 1
                
        # 儲存所有結果到 CSV 檔
        df = pd.DataFrame(all_results)

        # 指定想要的欄位順序，將 'experiment' 放在第一欄
        desired_columns = [
            "experiment",
            "model",
            "model_config",
            "run",
            "acc",
            "recall",
            "precision",
            "auc",
            "f1",
            "data_path"
        ]

        # 重新排欄位
        df = df[desired_columns]

        # 輸出 CSV
        df.to_csv("experiment_results.csv", index=False)
        
        return all_results
