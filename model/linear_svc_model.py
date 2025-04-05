from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from .base_model import BaseModel
 # Ensure that BaseModel is imported from your base_model.py file

class LinearSvcModel(BaseModel):
    """
    LinearSvcModel implements the BaseModel interface using a linear kernel SVC.
    """

    def __init__(self, params):
        '''
        四種不同SVC核函數:
            kernel='linear' (線性)
            kernel='poly' (非線性)
            kernel='rbf' (非線性)
            kernel='sigmoid' (非線性)
        C: 限制模型的複雜度, 防止過度擬合。
        max_iter: 最大迭代次數, 預設1000。
        '''
        self.config = params
        # Create an SVC instance with a linear kernel
        self.model = SVC(kernel=params.get('kernel', 'linear'), probability=True,
                         C=params.get('C', 1.0),
                         max_iter=params.get('max_iter', 3000))
        self.name = params.get('name', 'Linear SVC')


    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        self.train_predicted = self.model.predict(x)

    def predict_proba(self, x):
        self.train_predicted_prob = self.model.predict_proba(x)

    def evaluate(self, x, y):
        result = {
                "acc": round(self.model.score(x, y), 3),
            }
        return result

    def get_name(self):
        return self.get_name()
    
    # def set_train_data(self, x, y):
    #     self.train_data = x
    #     self.train_label = y

    # def set_test_data(self, x, y):
    #     self.test_data = x
    #     self.test_label = y