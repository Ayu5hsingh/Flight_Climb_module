#!/bin/bash

echo "=========================================="
echo "Running Flask API Tests"
echo "=========================================="
echo ""

# Check if Flask app is importable
echo "Checking Flask app..."
python -c "from app import app; print('✓ Flask app imports successfully')"

if [ $? -ne 0 ]; then
    echo "✗ Error: Cannot import Flask app. Make sure all dependencies are installed."
    exit 1
fi

echo ""
echo "Running unit tests..."
python test_app.py

echo ""
echo "=========================================="
echo "Tests Complete!"
echo "=========================================="