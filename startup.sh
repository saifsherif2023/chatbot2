#!/bin/bash

# Exit on any error
set -e

# Print commands as they are executed
set -x

# Ensure we're in the right directory
cd /home/site/wwwroot

echo "Current directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "Pip version: $(python3 -m pip --version)"

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Start the Flask app with Gunicorn
echo "Starting Gunicorn..."
gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 app:app