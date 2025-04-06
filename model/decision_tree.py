import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             roc_auc_score, roc_curve, auc, confusion_matrix)
from .base_model import BaseModel  # Ensure BaseModel is imported from the same package
from lib import plot, result

class DecisionTreeModel(BaseModel):
    """
    DecisionTreeModel implements the BaseModel interface using a Decision Tree classifier.
    """

    def __init__(self, params):
        """
        Initialize the Decision Tree model with the given parameters.

        Supported parameters:
            - criterion: Function to measure the quality of a split (default: 'gini').
            - max_depth: The maximum depth of the tree (default: None, meaning unlimited).
            - min_samples_split: The minimum number of samples required to split an internal node (default: 2).
            - name: Optional name for the model (default: 'Decision Tree').
        """
        self.config = params
        self.model = DecisionTreeClassifier(
            criterion=params.get('criterion', 'gini'),
            max_depth=params.get('max_depth', 5),
            splitter=params.get('splitter', 'best'),
            # min_samples_split=params.get('min_samples_split', 2)
        )
        self.name = params.get('name', 'Decision Tree')

    def train(self, x, y):
        """
        Train the Decision Tree model using the provided training data.
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
        Make predictions using the trained Decision Tree model.
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
