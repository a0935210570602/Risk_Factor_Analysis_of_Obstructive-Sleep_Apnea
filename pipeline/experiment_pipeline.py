import pandas as pd
from .model_config_parser import ModelConfigParser  # Import model configuration parser module
from .model_builder import ModelBuilder                # Import model builder module
from .data_processor import DataProcessor              # Import data processor module
from lib import balance_data, select_feature           # Import data balancing and feature selection modules

class ExperimentPipeline:
    def __init__(self, exp_config):
        """
        Initialize the experiment pipeline and store the experiment configurations.
        Args:
            exp_config (list): A list of experiment configurations, each containing data configuration,
                               model configuration, and optionally feature selection configuration.
        """
        self.exp_config = exp_config

    def data_process(self, data_config, feature_select_config):
        print("=== Starting data processing ===")
        # Set a default feature list as a fallback
        self.selected_feature_list = ['age', 'sex', 'HLOS', 'NIHSS', 'tPA(0/1)', 'EVT(0/1)',
                                       'HTN(0/1)', 'DM(0/1)', 'Dyslipidemia(0/1)', 'Af(0/1)',
                                       'smoking(Y/N/Q)', 'LDL ', 'cholesterol', 'TG', 'Cre',
                                       'SGPT', 'HbA1c', 'MRS']
        data_config["selected_feature_list"] = self.selected_feature_list

        # Create a DataProcessor instance and load data
        processor = DataProcessor(data_config)
        self.train_X, self.train_Y, self.test_X, self.test_Y = processor.load_data()
        print("Data loaded: train shape = {}, test shape = {}"
              .format(self.train_X.shape, self.test_X.shape))

        # Perform feature selection if configuration is provided
        if feature_select_config != "None":
            print("Starting feature selection with config:", feature_select_config)
            self.selected_feature_list = select_feature.feature_selection(self.train_X, self.train_Y, feature_select_config)
            print("Selected features:", self.selected_feature_list)
            # Update data_config with the selected features and reload the data
            data_config["selected_feature_list"] = self.selected_feature_list
            processor = DataProcessor(data_config)
            self.train_X, self.train_Y, self.test_X, self.test_Y = processor.load_data()
            print("After reloading data: train shape = {}, test shape = {}"
                  .format(self.train_X.shape, self.test_X.shape))
        else:
            print("No feature selection applied; using default features.")

        # Check if SMOTE data balancing is required
        if data_config.get("balance") == "smote":
            print("Applying SMOTE for data balancing.")
            self.train_X, self.train_Y = balance_data.smote(self.train_X, self.train_Y)
        print("=== Data processing completed ===\n")
    
    def model_process(self):
        print("=== Starting model processing ===")
        model_results = []
        for config in self.parsed_configs:
            runs = config["runs"]
            for run in range(runs):
                print("Training model: {} (Run {})".format(config["model_name"], run + 1))
                # Build model instance and train
                model_instance = ModelBuilder.build(config)
                model_instance.train(self.train_X, self.train_Y)
                model_instance.predict(self.train_X)
                train_evaluation = model_instance.evaluate(self.train_Y)
                model_instance.save_result()
                print("Training recall:", train_evaluation.get("recall"))

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

                # Predict and evaluate test set
                model_instance.predict(self.test_X)
                test_evaluation = model_instance.evaluate(self.test_Y)
                model_instance.save_result()
                print("Test recall:", test_evaluation.get("recall"))

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
        print("=== Model processing completed ===\n")
        return model_results

    def run(self):
        print("=== Starting experiment pipeline ===")
        all_results = []
        exp_num = 1

        for exp in self.exp_config:
            print(">>> Processing Experiment {} <<<".format(exp_num))
            data_config = exp.get("data_config", {})
            model_configs = exp.get("model_config", {})
            feature_select_config = exp.get("feature_select_config", "None")
            parser = ModelConfigParser(model_configs)
            self.parsed_configs = parser.parse()

            # Data processing: load and split data (with feature selection and balancing)
            self.data_process(data_config, feature_select_config)
            # Model processing: build, train, and evaluate models
            exp_results = self.model_process()

            # Append additional information to each result
            for res in exp_results:
                res["experiment"] = f"exp{exp_num}"
                res["data_path"] = data_config.get("path")
                res["balance"] = data_config.get("balance", None)
                res["feature_select_config"] = feature_select_config
                res["selected_features"] = self.selected_feature_list
                all_results.append(res)
            exp_num += 1
                
        # Save all results to a CSV file
        df = pd.DataFrame(all_results)
        desired_columns = [
            "experiment",
            "feature_select_config",
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
            "selected_features",
            "data_path",
        ]
        df = df[desired_columns]
        df.to_csv("experiment_results.csv", index=False)
        print("Experiment results saved to 'experiment_results.csv'")
        print("=== Experiment pipeline completed ===")
        return all_results
