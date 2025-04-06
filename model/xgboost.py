import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             roc_auc_score, roc_curve, auc, confusion_matrix)
from .base_model import BaseModel  # Ensure BaseModel is imported from the same package
from lib import plot, result

class XGBoostModel(BaseModel):
    """
    XGBoostModel implements the BaseModel interface using XGBoost.
    """

    def __init__(self, config):
        """
        Initialize the XGBoost model with the given configuration.

        Supported parameters:
            - n_estimators: Number of boosting rounds (default: 100).
            - max_depth: Maximum tree depth (default: None, meaning unlimited).
            - learning_rate: Boosting learning rate (default: 0.1).
            - objective: Learning objective (default: 'binary:logistic').
            - random_state: Seed used by the random number generator (default: None).
            - name: Optional name for the model (default: 'XGBoost').
        """
        self.config = config
        self.model = xgb.XGBClassifier(
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.1),
            scale_pos_weight=config.get('scale_pos_weight', 1),
            random_state=config.get('random_state', 42),
        )
        self.name = config.get('name', 'XGBoost')

    def train(self, x, y):
        print(f"訓練資料特徵維度: {x.shape}")
        print(f"訓練資料標籤維度: {y.shape}")
        
        # 統計標籤為 0 和 1 的數量
        unique, counts = np.unique(y, return_counts=True)
        label_counts = dict(zip(unique, counts))
        print("標籤統計:")
        for label, count in label_counts.items():
            print(f"標籤 {label}: {count} 筆資料")
        """
        Train the XGBoost model using the provided training data.
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
        Make predictions using the trained XGBoost model.
        """
        self.predictions = self.model.predict(x)
        self.probas = self.model.predict_proba(x)

    def evaluate(self, true_y):
        """
        Evaluate the performance of the model using the true labels.
        Returns a dictionary of evaluation metrics.
        """
        self.result = result.evaluate_model(true_y, self.predictions, self.probas)
        return self.result

    def save_result(self):
        """
        Save the evaluation results by plotting the confusion matrix and ROC curve.
        """
        plot.confusion_matrix(self.result['cm'], self.name, './result_data')
        plot.plot_ROC_curve(self.result['fpr'], self.result['tpr'], self.result['roc_auc'], self.name, './result_data')
