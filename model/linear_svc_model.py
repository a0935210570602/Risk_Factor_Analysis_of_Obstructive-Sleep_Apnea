import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score, roc_curve, auc,
                             confusion_matrix, precision_score)
from .base_model import BaseModel  # 確保 BaseModel 從同一個 package 正確匯入
from lib import plot, result
class LinearSvcModel(BaseModel):
    """
    LinearSvcModel implements the BaseModel interface using a linear kernel SVC.
    """

    def __init__(self, params):
        """
        Initialize the Linear SVC model with the given parameters.

        Supported SVC kernels:
            - 'linear' (線性)
            - 'poly' (非線性)
            - 'rbf' (非線性)
            - 'sigmoid' (非線性)
        C: Regularization parameter to prevent overfitting.
        max_iter: Maximum number of iterations, default is 3000.
        """
        self.config = params
        self.model = SVC(kernel=params.get('kernel', 'linear'),
                         probability=True,
                         C=params.get('C', 1.0),
                         max_iter=params.get('max_iter', 3000))
        self.name = params.get('name', 'Linear SVC')

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        self.predictions = self.model.predict(x)
        self.probas = self.model.predict_proba(x)

    def evaluate(self, true_y):
        self.result = result.evaluate_model(true_y, self.predictions, self.probas)
        return self.result

    def save_result(self):
        plot.confusion_matrix(self.result['cm'], self.name, './result_data')
        plot.plot_ROC_curve(self.result['fpr'], self.result['tpr'], self.result['roc_auc'], self.name, './result_data')