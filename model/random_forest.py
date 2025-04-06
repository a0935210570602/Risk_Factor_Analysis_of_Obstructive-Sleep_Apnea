import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             roc_auc_score, roc_curve, auc, confusion_matrix)
from .base_model import BaseModel  # Ensure BaseModel is imported from the same package
from lib import plot, result

class RandomForestModel(BaseModel):
    """
    RandomForestModel implements the BaseModel interface using a Random Forest classifier.
    """

    def __init__(self, params):
        """
        Initialize the Random Forest model with the given parameters.

        Supported parameters:
            - n_estimators: The number of trees in the forest (default: 100).
            - criterion: The function to measure the quality of a split (default: 'gini').
            - max_depth: The maximum depth of the tree (default: None, meaning unlimited).
            - random_state: The seed used by the random number generator (default: None).
            - name: Optional name for the model (default: 'Random Forest').
        """
        self.config = params
        self.model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            criterion=params.get('criterion', 'gini'),
            max_depth=params.get('max_depth', None),
            random_state=params.get('random_state', None)
        )
        self.name = params.get('name', 'Random Forest')

    def train(self, x, y):
        """
        Train the Random Forest model using the provided training data.
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
        Make predictions using the trained Random Forest model.
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
