#!/usr/bin/env python
"""
Integration tests for Flask API
These tests require the Flask server to be running
Run the server first: python app.py
Then run this script in another terminal: python integration_test.py
"""

import requests
import json
import sys


BASE_URL = "http://localhost:5050"


def test_endpoint(name, method, url, data=None, expected_status=200):
    """Helper function to test an endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    print(f"Method: {method}")
    print(f"URL: {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            print(f"❌ Unknown method: {method}")
            return False
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == expected_status:
            print(f"✓ Status code matches expected ({expected_status})")
        else:
            print(f"✗ Expected status {expected_status}, got {response.status_code}")
            return False
        
        # Try to parse JSON
        try:
            response_data = response.json()
            print(f"\nResponse (first 500 chars):")
            print(json.dumps(response_data, indent=2)[:500])
            print("✓ Response is valid JSON")
            return True
        except json.JSONDecodeError:
            print("✗ Response is not valid JSON")
            print(f"Response text: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed! Make sure the Flask server is running.")
        print("   Start the server with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Run all integration tests"""
    print("="*60)
    print("Flask API Integration Tests")
    print("="*60)
    print("\nMake sure the Flask server is running!")
    print("Start it with: python app.py")
    print("="*60)
    
    results = []
    
    # Test 1: Home route
    results.append(test_endpoint(
        "Home Route",
        "GET",
        f"{BASE_URL}/"
    ))
    
    # Test 2: Functions route
    results.append(test_endpoint(
        "Functions List",
        "GET",
        f"{BASE_URL}/functions"
    ))
    
    # Test 3: Process route (GET) - Classification
    results.append(test_endpoint(
        "Process Route (GET) - Classification",
        "GET",
        f"{BASE_URL}/process?n_samples=50&n_features=3&problem=classification"
    ))
    
    # Test 4: Process route (GET) - Regression
    results.append(test_endpoint(
        "Process Route (GET) - Regression",
        "GET",
        f"{BASE_URL}/process?n_samples=50&n_features=3&problem=regression"
    ))
    
    # Test 5: Process route (POST) - Classification
    results.append(test_endpoint(
        "Process Route (POST) - Classification",
        "POST",
        f"{BASE_URL}/process",
        data={"n_samples": 80, "n_features": 4, "problem": "classification"}
    ))
    
    # Test 6: Classification process route (GET)
    results.append(test_endpoint(
        "Classification Process (GET)",
        "GET",
        f"{BASE_URL}/classification/process?n_samples=60&n_features=5"
    ))
    
    # Test 7: Classification process route (POST)
    results.append(test_endpoint(
        "Classification Process (POST)",
        "POST",
        f"{BASE_URL}/classification/process",
        data={"n_samples": 70, "n_features": 6}
    ))
    
    # Test 8: Regression process route (GET)
    results.append(test_endpoint(
        "Regression Process (GET)",
        "GET",
        f"{BASE_URL}/regression/process?n_samples=60&n_features=5"
    ))
    
    # Test 9: Regression process route (POST)
    results.append(test_endpoint(
        "Regression Process (POST)",
        "POST",
        f"{BASE_URL}/regression/process",
        data={"n_samples": 70, "n_features": 6}
    ))
    
    # Test 10: Invalid problem type (should return 400)
    results.append(test_endpoint(
        "Invalid Problem Type (should fail with 400)",
        "GET",
        f"{BASE_URL}/process?problem=invalid_type",
        expected_status=400
    ))
    
    # Test 11: 404 error
    results.append(test_endpoint(
        "Non-existent Route (should return 404)",
        "GET",
        f"{BASE_URL}/nonexistent",
        expected_status=404
    ))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())