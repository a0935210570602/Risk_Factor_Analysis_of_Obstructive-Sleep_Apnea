# 建立一個解析 model_configs 的類別
class ModelConfigParser:
    def __init__(self, config_list):
        self.configs = config_list

    def parse(self):
        parsed_configs = []
        for config in self.configs:
            # 提取 model 名稱、參數與重複次數
            model_name = config.get("model_name")
            params = config.get("params", {})
            runs = config.get("runs", 1)
            # 封裝成一個字典（也可以封裝成自定義物件）
            parsed_config = {
                "model_name": model_name,
                "params": params,
                "runs": runs
            }
            parsed_configs.append(parsed_config)
        return parsed_configs

    def display(self):
        # 輸出所有解析結果
        for config in self.parse():
            print(f"Model: {config['name']}, Params: {config['params']}, Runs: {config['runs']}")

# 測試解析器
if __name__ == '__main__':
    parser = ModelConfigParser(model_configs)
    parser.display()
