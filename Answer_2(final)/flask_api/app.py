from flask import Flask, jsonify, request
import numpy as np
from  ml_model_sita_internship import (
    generate, learn, predict, statistics
)

app = Flask(__name__)

# Convert numpy types to Python native types for JSON serialization
def convert_to_serializable(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


@app.route('/', methods=['GET'])
def home():
    """
    Home route - returns a welcome message
    """
    return jsonify({
        "message": "Welcome to ML Model API!",
        "description": "A Flask API for machine learning classification and regression",
        "version": "0.1.0",
        "endpoints": [
            "/",
            "/functions",
            "/process",
            "/classification/process",
            "/regression/process"
        ]
    })


@app.route('/functions', methods=['GET'])
def list_functions():
    """
    Returns the list of all implemented functions
    """
    functions_list = [
        {
            "name": "generate",
            "description": "Generate a dataset for classification or regression",
            "parameters": ["problem", "n_samples", "n_features"]
        },
        {
            "name": "learn",
            "description": "Train a linear model on the dataset",
            "parameters": ["problem", "X", "y"]
        },
        {
            "name": "predict",
            "description": "Make predictions on new data",
            "parameters": ["model", "problem"]
        },
        {
            "name": "statistics",
            "description": "Compute descriptive statistics from the data",
            "parameters": ["X", "y"]
        },
        {
            "name": "get_metric",
            "description": "Get the appropriate metric for the problem type",
            "parameters": ["problem"]
        }
    ]
    
    return jsonify({
        "functions": functions_list,
        "total_functions": len(functions_list)
    })


@app.route('/process', methods=['POST', 'GET'])
def process():
    """
    Full ML pipeline: generate dataset, compute statistics, train model, make predictions
    
    Parameters (query params or JSON body):
    - n_samples: int (default: 100)
    - n_features: int (default: 5)
    - problem: str ("classification", "regression", or "ordinaryleastsquaresregression")
    """
    try:
        # Get parameters from query string or JSON body
        if request.method == 'POST' and request.is_json:
            data = request.get_json()
        else:
            data = request.args
        
        n_samples = int(data.get('n_samples', 100))
        n_features = int(data.get('n_features', 5))
        problem = data.get('problem', 'classification')
        
        # Validate problem type
        valid_problems = ['classification', 'regression', 'ordinaryleastsquaresregression']
        if problem not in valid_problems:
            return jsonify({
                "error": f"Invalid problem type. Must be one of: {valid_problems}"
            }), 400
        
        # Generate dataset
        X, y = generate(problem, n_samples=n_samples, n_features=n_features)
        
        # Compute statistics
        stats = statistics(X, y)
        
        # Train model
        model, error = learn(problem, X, y)
        
        # Make predictions
        predictions = predict(model, problem)
        
        # Prepare response
        response = {
            "problem_type": problem,
            "n_samples": n_samples,
            "n_features": n_features,
            "statistics": convert_to_serializable(stats),
            "model_error": float(error),
            "predictions": convert_to_serializable(predictions),
            "metric_type": "accuracy" if problem == "classification" else "mean_squared_error"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred during processing"
        }), 500


@app.route('/classification/process', methods=['POST', 'GET'])
def classification_process():
    """
    Classification-specific pipeline
    
    Parameters (query params or JSON body):
    - n_samples: int (default: 100)
    - n_features: int (default: 5)
    """
    try:
        # Get parameters
        if request.method == 'POST' and request.is_json:
            data = request.get_json()
        else:
            data = request.args
        
        n_samples = int(data.get('n_samples', 100))
        n_features = int(data.get('n_features', 5))
        problem = 'classification'
        
        # Generate dataset
        X, y = generate(problem, n_samples=n_samples, n_features=n_features)
        
        # Compute statistics
        stats = statistics(X, y)
        
        # Train model
        model, error = learn(problem, X, y)
        
        # Make predictions
        predictions = predict(model, problem)
        
        # Prepare response
        response = {
            "problem_type": problem,
            "n_samples": n_samples,
            "n_features": n_features,
            "statistics": convert_to_serializable(stats),
            "accuracy": float(error),
            "predictions": convert_to_serializable(predictions)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred during classification processing"
        }), 500


@app.route('/regression/process', methods=['POST', 'GET'])
def regression_process():
    """
    Regression-specific pipeline
    
    Parameters (query params or JSON body):
    - n_samples: int (default: 100)
    - n_features: int (default: 5)
    """
    try:
        # Get parameters
        if request.method == 'POST' and request.is_json:
            data = request.get_json()
        else:
            data = request.args
        
        n_samples = int(data.get('n_samples', 100))
        n_features = int(data.get('n_features', 5))
        problem = 'regression'
        
        # Generate dataset
        X, y = generate(problem, n_samples=n_samples, n_features=n_features)
        
        # Compute statistics
        stats = statistics(X, y)
        
        # Train model
        model, error = learn(problem, X, y)
        
        # Make predictions
        predictions = predict(model, problem)
        
        # Prepare response
        response = {
            "problem_type": problem,
            "n_samples": n_samples,
            "n_features": n_features,
            "statistics": convert_to_serializable(stats),
            "mean_squared_error": float(error),
            "predictions": convert_to_serializable(predictions)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred during regression processing"
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested URL was not found on the server"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500


if __name__ == '__main__':
    # Run the Flask app
    # debug=True enables auto-reload and better error messages
    app.run(debug=True, host='0.0.0.0', port=5050)