# model_builder.py

from model.linear_svc import LinearSvcModel
from model.decision_tree import DecisionTreeModel
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
        # elif model_name == "xgb":
        #     from .xgb_model import XGBModel
        #     return XGBModel(params)
        elif model_name == "decision_tree":
            return DecisionTreeModel(params)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
