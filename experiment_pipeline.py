# experiment_pipeline.py
from model_config_parser import ModelConfigParser
from model_builder import ModelBuilder

class ExperimentPipeline:
    def __init__(self, model_configs, train_X, train_Y, test_X, test_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        
        # 解析模型配置
        parser = ModelConfigParser(model_configs)
        self.parsed_configs = parser.parse()
    
    def run(self):
        results = []
        for config in self.parsed_configs:
            runs = config["runs"]
            for run in range(runs):
                # 使用 ModelBuilder 根據配置建立模型
                model_instance = ModelBuilder.build(config)
                # 訓練模型
                model_instance.train(self.train_X, self.train_Y)
                # 評估模型（例如返回準確率）
                evaluate = model_instance.evaluate(self.test_X, self.test_Y)
                results.append({
                    "model": config["model_name"],
                    "run": run + 1,
                    "result": evaluate
                })
        return results
