#!/bin/bash
# Start Qdrant vector database using Docker

set -e

echo "Starting Qdrant..."
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    echo ""
    echo "Please install Docker from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "ERROR: Docker daemon is not running"
    echo ""
    echo "Please start Docker Desktop first, then run this script again"
    exit 1
fi

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^qdrant$"; then
    echo "Qdrant container found. Starting it..."
    docker start qdrant
else
    echo "Creating new Qdrant container..."
    docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant
fi

echo ""
echo "Qdrant is running!"
echo ""
echo "API: http://localhost:6333"
echo "Dashboard: http://localhost:6334/dashboard"
echo ""
echo "To stop: docker stop qdrant"
echo "To remove: docker rm qdrant"
echo ""
