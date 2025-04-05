from model_config_parser import ModelConfigParser
from model_builder import ModelBuilder
from data_processor import DataProcessor

class ExperimentPipeline:
    def __init__(self, exp_config):
        self.exp_config = exp_config

    def data_process(self, data_config):
        processor = DataProcessor(data_config)
        return processor.load_data()

    def model_process(self, model_configs, train_X, train_Y, test_X, test_Y):
        model_results = []
        for config in self.parsed_configs:
            runs = config["runs"]
            for run in range(runs):
                # Build model instance according to configuration
                model_instance = ModelBuilder.build(config)
                # Train model
                model_instance.train(train_X, train_Y)
                # Evaluate model (e.g., accuracy)
                evaluation = model_instance.evaluate(test_X, test_Y)
                model_results.append({
                    "model": config["model_name"],
                    "run": run + 1,
                    "result": evaluation
                })
        return model_results

    def run(self):
        results = []
        for exp in self.exp_config:
            data_config = exp.get("data_config", {})
            model_configs = exp.get("model_config", {})
            parser = ModelConfigParser(model_configs)
            self.parsed_configs = parser.parse()
            # Data processing: load and split data
            train_X, train_Y, test_X, test_Y = self.data_process(data_config)
            # Model processing: build, train, evaluate models
            exp_results = self.model_process(model_configs, train_X, train_Y, test_X, test_Y)
            
            # 為每個結果添加數據路徑信息
            for res in exp_results:
                res["data_path"] = data_config.get("path")
                results.append(res)
                
        return results
