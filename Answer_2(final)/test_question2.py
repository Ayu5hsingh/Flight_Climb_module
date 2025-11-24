import unittest
import numpy as np
from ml_model_sita_internship  import (
    generate, learn, predict, statistics,
    target_statistics, features_statistics, 
    correlation, get_metric
)
from sklearn.metrics import mean_squared_error, accuracy_score


class TestGenerate(unittest.TestCase):
    """Test dataset generation functions"""
    
    def test_classification_generation(self):
        """Test if classification dataset is generated correctly"""
        X, y = generate("classification", n_samples=100, n_features=5)
        
        # Check shapes
        self.assertEqual(X.shape, (100, 5))
        self.assertEqual(y.shape, (100,))
        
        # Check if binary classification (only 0 and 1)
        self.assertTrue(set(np.unique(y)).issubset({0, 1}))
    
    def test_regression_generation(self):
        """Test if regression dataset is generated correctly"""
        X, y = generate("regression", n_samples=50, n_features=3)
        
        # Check shapes
        self.assertEqual(X.shape, (50, 3))
        self.assertEqual(y.shape, (50,))
        
        # Check if continuous values
        self.assertTrue(np.issubdtype(y.dtype, np.floating))
    
    def test_ols_regression_generation(self):
        """Test OLS regression dataset generation"""
        X, y = generate("ordinaryleastsquaresregression", n_samples=80, n_features=4)
        
        self.assertEqual(X.shape, (80, 4))
        self.assertEqual(y.shape, (80,))
    
    def test_invalid_problem_type(self):
        """Test if invalid problem type raises error"""
        with self.assertRaises(Exception):
            generate("invalid_type", n_samples=10, n_features=2)


class TestMetrics(unittest.TestCase):
    """Test metric functions"""
    
    def test_classification_metric(self):
        """Test if correct metric is returned for classification"""
        metric = get_metric("classification")
        self.assertEqual(metric, accuracy_score)
    
    def test_regression_metric(self):
        """Test if correct metric is returned for regression"""
        metric = get_metric("regression")
        self.assertEqual(metric, mean_squared_error)
    
    def test_ols_metric(self):
        """Test if correct metric is returned for OLS"""
        metric = get_metric("ordinaryleastsquaresregression")
        self.assertEqual(metric, mean_squared_error)


class TestStatistics(unittest.TestCase):
    """Test statistics functions"""
    
    def setUp(self):
        """Create a simple dataset for testing"""
        self.X = np.array([[1, 2], [3, 4], [5, 6]])
        self.y = np.array([1, 2, 3])
    
    def test_target_statistics(self):
        """Test target statistics calculation"""
        mean, std = target_statistics(self.y)
        
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(std, np.std([1, 2, 3]))
    
    def test_features_statistics(self):
        """Test feature statistics calculation"""
        mean_features, std_features = features_statistics(self.X)
        
        # Check if returns dictionaries
        self.assertIsInstance(mean_features, dict)
        self.assertIsInstance(std_features, dict)
        
        # Check number of features
        self.assertEqual(len(mean_features), 2)
        self.assertEqual(len(std_features), 2)
        
        # Check values
        self.assertAlmostEqual(mean_features['feature_0'], 3.0)
        self.assertAlmostEqual(mean_features['feature_1'], 4.0)
    
    def test_correlation(self):
        """Test correlation calculation"""
        corr_coefs = correlation(self.X, self.y)
        
        # Check if returns dictionary
        self.assertIsInstance(corr_coefs, dict)
        
        # Check number of features
        self.assertEqual(len(corr_coefs), 2)
    
    def test_full_statistics(self):
        """Test complete statistics function"""
        stats = statistics(self.X, self.y)
        
        # Check if all keys are present
        required_keys = ['mean_target', 'std_target', 'mean_features', 
                        'std_features', 'correlations']
        for key in required_keys:
            self.assertIn(key, stats)


class TestModelTraining(unittest.TestCase):
    """Test model training functions"""
    
    def test_classification_training(self):
        """Test if classification model trains successfully"""
        X, y = generate("classification", n_samples=100, n_features=5)
        model, error = learn("classification", X, y)
        
        # Check if model is trained
        self.assertIsNotNone(model)
        
        # Check if error is a valid number
        self.assertIsInstance(error, (int, float))
        self.assertGreaterEqual(error, 0.0)
        self.assertLessEqual(error, 1.0)  # Accuracy should be between 0 and 1
    
    def test_regression_training(self):
        """Test if regression model trains successfully"""
        X, y = generate("regression", n_samples=100, n_features=5)
        model, error = learn("regression", X, y)
        
        # Check if model is trained
        self.assertIsNotNone(model)
        
        # Check if error is a valid number (MSE)
        self.assertIsInstance(error, (int, float))
        self.assertGreaterEqual(error, 0.0)
    
    def test_ols_training(self):
        """Test if OLS model trains successfully"""
        X, y = generate("ordinaryleastsquaresregression", n_samples=100, n_features=5)
        model, error = learn("ordinaryleastsquaresregression", X, y)
        
        # Check if model is trained
        self.assertIsNotNone(model)
        self.assertGreaterEqual(error, 0.0)


class TestPrediction(unittest.TestCase):
    """Test prediction functions"""
    
    def test_classification_prediction(self):
        """Test if classification predictions work"""
        X, y = generate("classification", n_samples=100, n_features=5)
        model, _ = learn("classification", X, y)
        predictions = predict(model, "classification")
        
        # Check if predictions have correct length
        self.assertEqual(len(predictions), 10)
        
        # Check if predictions are binary
        self.assertTrue(set(np.unique(predictions)).issubset({0, 1}))
    
    def test_regression_prediction(self):
        """Test if regression predictions work"""
        X, y = generate("regression", n_samples=100, n_features=5)
        model, _ = learn("regression", X, y)
        predictions = predict(model, "regression")
        
        # Check if predictions have correct length
        self.assertEqual(len(predictions), 10)
        
        # Check if predictions are continuous
        self.assertTrue(np.issubdtype(predictions.dtype, np.floating))
    
    def test_ols_prediction(self):
        """Test if OLS predictions work"""
        X, y = generate("ordinaryleastsquaresregression", n_samples=100, n_features=5)
        model, _ = learn("ordinaryleastsquaresregression", X, y)
        predictions = predict(model, "ordinaryleastsquaresregression")
        
        # Check if predictions have correct length
        self.assertEqual(len(predictions), 10)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)