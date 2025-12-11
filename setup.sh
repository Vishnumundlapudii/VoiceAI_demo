#!/bin/bash

echo "======================================"
echo "E2E Voice Assistant - Pipecat Setup"
echo "======================================"

# Create virtual environment
echo "1. Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "2. Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "3. Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "4. Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To run the assistant:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run server: python server.py"
echo "3. Open browser: http://localhost:8080"
echo ""
echo "To test locally:"
echo "python test_local.py"