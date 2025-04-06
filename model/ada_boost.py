import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             roc_auc_score, roc_curve, auc, confusion_matrix)
from .base_model import BaseModel  # Ensure BaseModel is imported from the same package
from lib import plot, result

class AdaBoostModel(BaseModel):
    """
    AdaBoostModel implements the BaseModel interface using AdaBoost.
    """
    
    def __init__(self, config):
        """
        Initialize the AdaBoost model with the given configuration.
        
        Supported parameters:
            - n_estimators: The maximum number of estimators at which boosting is terminated (default: 50).
            - learning_rate: Weight applied to each classifier at each boosting iteration (default: 1.0).
            - algorithm: Algorithm used to update weights after each boosting iteration (default: 'SAMME.R').
            - random_state: Seed for the random number generator (default: None).
            - name: Optional name for the model (default: 'AdaBoost').
        """
        self.config = config
        self.model = AdaBoostClassifier(
            n_estimators=config.get('n_estimators', 50),
            learning_rate=config.get('learning_rate', 1.0),
            algorithm=config.get('algorithm', 'SAMME.R'),
            random_state=config.get('random_state', None)
        )
        self.name = config.get('name', 'AdaBoost')

    def train(self, x, y):
        """
        Train the AdaBoost model using the provided training data.
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
        Make predictions using the trained AdaBoost model.
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
