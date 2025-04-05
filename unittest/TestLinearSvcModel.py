import unittest
import numpy as np
from sklearn.datasets import make_classification
from model import BaseModel  # Adjust the import path as necessary
from model import LinearSvcModel  # Adjust the import path as necessary

class TestLinearSvcModel(unittest.TestCase):
    def setUp(self):
        """
        Set up synthetic training and test datasets for binary classification.
        """
        # Generate a binary classification dataset for training
        self.X_train, self.y_train = make_classification(
            n_samples=100, n_features=20, n_informative=5, n_redundant=0,
            n_classes=2, random_state=42
        )
        # Generate a separate dataset for testing
        self.X_test, self.y_test = make_classification(
            n_samples=30, n_features=20, n_informative=5, n_redundant=0,
            n_classes=2, random_state=43
        )
        # Define configuration for the model
        self.config = {"C": 1.0, "max_iter": 3000}

    def test_training_and_prediction(self):
        """
        Test that the model can be trained and makes predictions.
        """
        model = LinearSvcModel(self.config)
        model.train(self.X_train, self.y_train)
        
        # Test predict: output should have the same number of samples as X_test
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), self.X_test.shape[0])
        
        # Test predict_proba: the output should be of shape (n_samples, n_classes)
        probabilities = model.predict_proba(self.X_test)
        self.assertEqual(probabilities.shape[0], self.X_test.shape[0])
        self.assertEqual(probabilities.shape[1], 2)  # For binary classification

        # Each row in predict_proba output should sum approximately to 1
        for prob in probabilities:
            self.assertAlmostEqual(np.sum(prob), 1.0, places=5)

    def test_evaluation(self):
        """
        Test that the evaluate method returns a valid accuracy score.
        """
        model = LinearSvcModel(self.config)
        model.train(self.X_train, self.y_train)
        accuracy = model.evaluate(self.X_test, self.y_test)
        
        # Check that accuracy is a float and lies between 0 and 1.
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

if __name__ == "__main__":
    unittest.main()
