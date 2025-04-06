import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             roc_auc_score, roc_curve, auc, confusion_matrix)
from .base_model import BaseModel  # Ensure BaseModel is imported from the same package
from lib import plot, result

class GradientBoostModel(BaseModel):
    """
    GradientBoostModel implements the BaseModel interface using Gradient Boosting.
    """

    def __init__(self, config):
        """
        Initialize the Gradient Boosting model with the given configuration.

        Supported parameters:
            - n_estimators: The number of boosting stages to perform (default: 100).
            - learning_rate: Shrinks the contribution of each tree (default: 0.1).
            - max_depth: Maximum depth of the individual regression estimators (default: 3).
            - random_state: Seed for the random number generator (default: None).
            - name: Optional name for the model (default: 'Gradient Boosting').
        """
        self.config = config
        self.model = GradientBoostingClassifier(
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.1),
            max_depth=config.get('max_depth', 3),
            random_state=config.get('random_state', None)
        )
        self.name = config.get('name', 'Gradient Boosting')

    def train(self, x, y):
        """
        Train the Gradient Boosting model using the provided training data.
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
        Make predictions using the trained Gradient Boosting model.
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
