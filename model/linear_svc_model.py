import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve, auc
from .base_model import BaseModel  # 確保 BaseModel 從同一個 package 正確匯入

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
        self.train_predicted = self.model.predict(x)
        return self.train_predicted

    def predict_proba(self, x):
        self.train_predicted_prob = self.model.predict_proba(x)
        return self.train_predicted_prob

    def evaluate(self, x, y, save_path='./5.png', title="ROC Curve"):
        """
        Evaluate the model using multiple metrics: accuracy, recall,
        ROC AUC, as well as the ROC curve's false positive rate (fpr) and true positive rate (tpr).
        If save_path is provided, the ROC curve will be plotted and saved to the specified path.

        Args:
            x (array-like): Test data features.
            y (array-like): True labels.
            save_path (str, optional): File path to save the ROC curve image.
            title (str, optional): Title for the ROC curve plot.

        Returns:
            dict: A dictionary containing:
                - "acc": Accuracy score (rounded to 3 decimals).
                - "recall": Recall score (rounded to 3 decimals).
                - "roc_auc": ROC AUC score (rounded to 3 decimals) computed directly.
                - "fpr": False positive rates (array).
                - "tpr": True positive rates (array).
                - "auc": Area under the ROC curve computed from fpr/tpr (rounded to 3 decimals).
        """
        # 取得預測結果與預測機率（正類概率）
        predictions = self.model.predict(x)
        probas = self.model.predict_proba(x)[:, 1]

        # 計算各項指標
        acc = round(accuracy_score(y, predictions), 3)
        rec = round(recall_score(y, predictions), 3)
        roc_auc_direct = round(roc_auc_score(y, probas), 3)
        
        # 計算 ROC 曲線
        fpr, tpr, thresholds = roc_curve(y, probas)
        auc_value = round(auc(fpr, tpr), 3)

        # 若提供 save_path，則繪製並儲存 ROC 曲線圖
        if save_path:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_value})')
            plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(save_path)
            plt.show()
            # plt.close()

        result = {
            "acc": acc,
            "recall": rec,
            "roc_auc": roc_auc_direct,
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc_value
        }
        return result

    def get_name(self):
        return self.name
