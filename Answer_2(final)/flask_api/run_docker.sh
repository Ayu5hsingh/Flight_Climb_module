#!/bin/bash

echo "=========================================="
echo "Building ML Model Docker Image"
echo "=========================================="

# Check if the package exists
if [ ! -f "dist/ml_model_sita_internship-0.1.0.tar.gz" ]; then
    echo "Error: ML package not found!"
    echo "Please copy dist/ml_model_sita_internship-0.1.0.tar.gz to this directory"
    exit 1
fi

echo ""
echo "Building development Docker image..."
docker build -t ml-model-api:dev -f Dockerfile .

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Development image built successfully!"
    echo "Image name: ml-model-api:dev"
else
    echo ""
    echo "✗ Build failed!"
    exit 1
fi

echo ""
echo "Building production Docker image..."
docker build -t ml-model-api:prod -f Dockerfile.prod .

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Production image built successfully!"
    echo "Image name: ml-model-api:prod"
else
    echo ""
    echo "✗ Build failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Docker Images Built Successfully!"
echo "=========================================="
echo ""
echo "Available images:"
docker images | grep ml-model-api
echo ""
echo "To run the development image:"
echo "  docker run -p 5000:5000 ml-model-api:dev"
echo ""
echo "To run the production image:"
echo "  docker run -p 5000:5000 ml-model-api:prod"
echo ""
echo "Or use docker-compose:"
echo "  docker-compose up"