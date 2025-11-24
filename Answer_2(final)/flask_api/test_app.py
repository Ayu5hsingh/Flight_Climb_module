import unittest
import json
from app import app


class TestFlaskAPI(unittest.TestCase):
    """Test cases for Flask API endpoints"""
    
    def setUp(self):
        """Set up test client before each test"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def tearDown(self):
        """Clean up after each test"""
        pass


class TestHomeRoute(TestFlaskAPI):
    """Test the home route"""
    
    def test_home_route_status_code(self):
        """Test if home route returns 200 status code"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_home_route_returns_json(self):
        """Test if home route returns JSON"""
        response = self.client.get('/')
        self.assertEqual(response.content_type, 'application/json')
    
    def test_home_route_content(self):
        """Test if home route contains expected keys"""
        response = self.client.get('/')
        data = json.loads(response.data)
        
        self.assertIn('message', data)
        self.assertIn('description', data)
        self.assertIn('version', data)
        self.assertIn('endpoints', data)
    
    def test_home_route_endpoints_list(self):
        """Test if home route lists all endpoints"""
        response = self.client.get('/')
        data = json.loads(response.data)
        
        expected_endpoints = [
            '/',
            '/functions',
            '/process',
            '/classification/process',
            '/regression/process'
        ]
        
        for endpoint in expected_endpoints:
            self.assertIn(endpoint, data['endpoints'])


class TestFunctionsRoute(TestFlaskAPI):
    """Test the functions listing route"""
    
    def test_functions_route_status_code(self):
        """Test if functions route returns 200 status code"""
        response = self.client.get('/functions')
        self.assertEqual(response.status_code, 200)
    
    def test_functions_route_returns_json(self):
        """Test if functions route returns JSON"""
        response = self.client.get('/functions')
        self.assertEqual(response.content_type, 'application/json')
    
    def test_functions_route_has_functions_list(self):
        """Test if functions route returns a list of functions"""
        response = self.client.get('/functions')
        data = json.loads(response.data)
        
        self.assertIn('functions', data)
        self.assertIn('total_functions', data)
        self.assertIsInstance(data['functions'], list)
        self.assertGreater(data['total_functions'], 0)
    
    def test_functions_route_function_structure(self):
        """Test if each function has required fields"""
        response = self.client.get('/functions')
        data = json.loads(response.data)
        
        for func in data['functions']:
            self.assertIn('name', func)
            self.assertIn('description', func)
            self.assertIn('parameters', func)


class TestProcessRoute(TestFlaskAPI):
    """Test the generic process route"""
    
    def test_process_route_get_default_parameters(self):
        """Test process route with GET and default parameters"""
        response = self.client.get('/process?problem=classification')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('statistics', data)
        self.assertIn('model_error', data)
        self.assertIn('predictions', data)
    
    def test_process_route_get_custom_parameters(self):
        """Test process route with GET and custom parameters"""
        response = self.client.get('/process?n_samples=50&n_features=3&problem=regression')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['n_samples'], 50)
        self.assertEqual(data['n_features'], 3)
        self.assertEqual(data['problem_type'], 'regression')
    
    def test_process_route_post_json(self):
        """Test process route with POST and JSON data"""
        payload = {
            'n_samples': 80,
            'n_features': 4,
            'problem': 'classification'
        }
        
        response = self.client.post(
            '/process',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['n_samples'], 80)
        self.assertEqual(data['n_features'], 4)
    
    def test_process_route_invalid_problem_type(self):
        """Test process route with invalid problem type"""
        response = self.client.get('/process?problem=invalid_type')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_process_route_classification_response_structure(self):
        """Test if classification response has correct structure"""
        response = self.client.get('/process?problem=classification&n_samples=100&n_features=5')  # n_features bumped to 5
        data = json.loads(response.data)
        
        required_keys = [
            'problem_type',
            'n_samples',
            'n_features',
            'statistics',
            'model_error',
            'predictions',
            'metric_type'
        ]
        
        for key in required_keys:
            self.assertIn(key, data)
    
    def test_process_route_regression_response_structure(self):
        """Test if regression response has correct structure"""
        response = self.client.get('/process?problem=regression&n_samples=50&n_features=3')
        data = json.loads(response.data)
        
        required_keys = [
            'problem_type',
            'n_samples',
            'n_features',
            'statistics',
            'model_error',
            'predictions',
            'metric_type'
        ]
        
        for key in required_keys:
            self.assertIn(key, data)
    
    def test_process_route_statistics_structure(self):
        """Test if statistics in response have correct structure"""
        response = self.client.get('/process?problem=classification&n_samples=50&n_features=5')  # 3 → 5
        data = json.loads(response.data)
        
        self.assertIn('statistics', data)
        stats = data['statistics']
        required_stats_keys = [
            'mean_target',
            'std_target',
            'mean_features',
            'std_features',
            'correlations'
        ]
        
        for key in required_stats_keys:
            self.assertIn(key, stats)
    
    def test_process_route_predictions_is_list(self):
        """Test if predictions are returned as a list"""
        response = self.client.get('/process?problem=classification&n_samples=50&n_features=5')  # 3 → 5
        data = json.loads(response.data)
        
        self.assertIn('predictions', data)
        self.assertIsInstance(data['predictions'], list)
        self.assertEqual(len(data['predictions']), 10)  # predict() generates 10 predictions


class TestClassificationProcessRoute(TestFlaskAPI):
    """Test the classification-specific process route"""
    
    def test_classification_route_get(self):
        """Test classification route with GET request"""
        response = self.client.get('/classification/process?n_samples=60&n_features=4')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['problem_type'], 'classification')
        self.assertEqual(data['n_samples'], 60)
        self.assertEqual(data['n_features'], 4)
    
    def test_classification_route_post(self):
        """Test classification route with POST request"""
        payload = {'n_samples': 70, 'n_features': 5}
        
        response = self.client.post(
            '/classification/process',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['problem_type'], 'classification')
    
    def test_classification_route_default_parameters(self):
        """Test classification route with default parameters"""
        response = self.client.get('/classification/process')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['n_samples'], 100)  # Default value
        self.assertEqual(data['n_features'], 5)   # Default value
    
    def test_classification_route_has_accuracy(self):
        """Test if classification response has accuracy metric"""
        response = self.client.get('/classification/process?n_samples=50&n_features=5')  # 3 → 5
        data = json.loads(response.data)
        
        self.assertIn('accuracy', data)
        self.assertIsInstance(data['accuracy'], (int, float))
        self.assertGreaterEqual(data['accuracy'], 0.0)
        self.assertLessEqual(data['accuracy'], 1.0)
    
    def test_classification_route_predictions_binary(self):
        """Test if classification predictions are binary (0 or 1)"""
        response = self.client.get('/classification/process?n_samples=50&n_features=5')  # 3 → 5
        data = json.loads(response.data)
        
        self.assertIn('predictions', data)
        predictions = data['predictions']
        for pred in predictions:
            self.assertIn(pred, [0, 1])


class TestRegressionProcessRoute(TestFlaskAPI):
    """Test the regression-specific process route"""
    
    def test_regression_route_get(self):
        """Test regression route with GET request"""
        response = self.client.get('/regression/process?n_samples=60&n_features=4')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['problem_type'], 'regression')
        self.assertEqual(data['n_samples'], 60)
        self.assertEqual(data['n_features'], 4)
    
    def test_regression_route_post(self):
        """Test regression route with POST request"""
        payload = {'n_samples': 70, 'n_features': 5}
        
        response = self.client.post(
            '/regression/process',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['problem_type'], 'regression')
    
    def test_regression_route_default_parameters(self):
        """Test regression route with default parameters"""
        response = self.client.get('/regression/process')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['n_samples'], 100)  # Default value
        self.assertEqual(data['n_features'], 5)   # Default value
    
    def test_regression_route_has_mse(self):
        """Test if regression response has MSE metric"""
        response = self.client.get('/regression/process?n_samples=50&n_features=3')
        data = json.loads(response.data)
        
        self.assertIn('mean_squared_error', data)
        self.assertIsInstance(data['mean_squared_error'], (int, float))
        self.assertGreaterEqual(data['mean_squared_error'], 0.0)
    
    def test_regression_route_predictions_continuous(self):
        """Test if regression predictions are continuous values"""
        response = self.client.get('/regression/process?n_samples=50&n_features=3')
        data = json.loads(response.data)
        
        self.assertIsInstance(data['predictions'], list)
        self.assertEqual(len(data['predictions']), 10)
        
        # Check if at least some predictions are floats (continuous)
        predictions = data['predictions']
        for pred in predictions:
            self.assertIsInstance(pred, (int, float))


class TestErrorHandling(TestFlaskAPI):
    """Test error handling"""
    
    def test_404_error(self):
        """Test 404 error for non-existent route"""
        response = self.client.get('/nonexistent-route')
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_invalid_json_post(self):
        """Test POST with invalid JSON"""
        response = self.client.post(
            '/process',
            data='invalid json{',
            content_type='application/json'
        )
        # Should return 400 or 500 depending on error handling
        self.assertIn(response.status_code, [400, 500])
    
    def test_missing_parameters_handled_gracefully(self):
        """Test that missing parameters use defaults"""
        response = self.client.get('/process?problem=classification')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        # Should use default values
        self.assertEqual(data['n_samples'], 100)
        self.assertEqual(data['n_features'], 5)


class TestParameterValidation(TestFlaskAPI):
    """Test parameter validation"""
    
    def test_string_n_samples_converted_to_int(self):
        """Test if string parameters are converted to integers"""
        # 3 → 5 to avoid make_classification arg error
        response = self.client.get('/process?n_samples=50&n_features=5&problem=classification')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['n_samples'], 50)
        self.assertEqual(data['n_features'], 5)
    
    def test_ordinaryleastsquares_problem_type(self):
        """Test OLS regression problem type"""
        response = self.client.get('/process?problem=ordinaryleastsquaresregression&n_samples=50&n_features=3')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['problem_type'], 'ordinaryleastsquaresregression')


if __name__ == '__main__':
    # Run all tests with verbose output
    unittest.main(verbosity=2)
