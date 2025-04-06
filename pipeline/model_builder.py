# model_builder.py

from model.linear_svc import LinearSvcModel
from model.decision_tree import DecisionTreeModel
from model.random_forest import RandomForestModel
from model.xgboost import XGBoostModel
from model.ada_boost import AdaBoostModel
from model.gradient_boost import GradientBoostModel
# 如果有其他模型，也可在此處引入，例如：
# from .xgb_model import XGBModel
# from .decision_tree_model import DecisionTreeModel

class ModelBuilder:

    @staticmethod
    def build(config):
        model_name = config.get("model_name")
        params = config.get("params", {})

        if model_name == "svm_linear":
            return LinearSvcModel(params)
        # 以下可以擴充其他模型
        elif model_name == "xgboost":
            return XGBoostModel(params)
        elif model_name == "decision_tree":
            return DecisionTreeModel(params)
        elif model_name == "random_forest":
            return DecisionTreeModel(params)
        elif model_name == "adaboost":
            return AdaBoostModel(params)
        elif model_name == "gradient_boost":
            return GradientBoostModel(params)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
