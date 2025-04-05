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
        Predict the output (class labels or continuous values) for the given input data.

        Args:
            x (array-like): Input features for which predictions are to be made.
            y (array-like): (Optional) Ground truth labels; may be used for special cases 
                            or further processing. In many cases, this can be ignored.

        Returns:
            array-like: Predicted outputs (e.g., class labels or regression values).
        """
        pass

    @abstractmethod
    def predict_proba(self):
        """
        Predict the probability estimates for each class given the input data.

        Args:
            x (array-like): Input features for which probability estimates are desired.
            y (array-like): (Optional) Ground truth labels; may be used for special processing.
        
        Returns:
            array-like: An array of probability estimates for each class for every input sample.
                        For binary classification, typically a shape of (n_samples, 2) is expected.
        """
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

    @abstractmethod
    def get_name(self):
        """
        Get the name of the model.

        Returns:
            str: The name of the model.
        """
        pass
    
    # @abstractmethod
    # def set_train_data(self, x, y):
    #     pass

    # @abstractmethod
    # def set_test_data(self, x, y):
    #     pass
