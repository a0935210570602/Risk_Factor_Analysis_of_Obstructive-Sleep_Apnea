from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def __init__(self, config):
        """
        Initialize the model with the given configuration.

        Args:
            config (dict): A dictionary containing configuration parameters 
                           required to initialize the model (e.g., hyperparameters, 
                           file paths, etc.).
        """
        pass

    @abstractmethod
    def train(self):
        """
        Train the model using the provided training data.

        Args:
            x (array-like): Training data features.
            y (array-like): Training data labels.

        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Make predictions using the trained model.

        Args:
            X (array-like): Input data features for making predictions.

        Returns:
            array-like: Predicted labels or probabilities for the input data.
        """
        pass
    
    @abstractmethod
    def save_result(self):
        pass
    
    @abstractmethod
    def evaluate(self):
        """
        Evaluate the performance of the model using the provided test data.

        Args:
            X (array-like): Test data features.
            y (array-like): True labels for the test data.

        Returns:
            float or dict: A performance metric (e.g., accuracy score) or a dictionary of metrics 
                           summarizing the model's performance on the test set.
        """
        pass