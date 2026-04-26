#!/bin/bash

# Video Captioning System - Quick Start Script

echo "🎬 Video Captioning System - Quick Start"
echo "========================================"

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.10+ first."
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed successfully"

# Create dummy dataset
echo "📁 Creating dummy dataset..."
python -m src.cli create-dataset --num-samples 10

if [ $? -ne 0 ]; then
    echo "❌ Failed to create dataset"
    exit 1
fi

echo "✅ Dataset created successfully"

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v

if [ $? -ne 0 ]; then
    echo "⚠️  Some tests failed, but continuing..."
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Launch the demo: streamlit run demo/streamlit_app.py"
echo "2. Train a model: python -m src.cli train --num-epochs 5"
echo "3. Generate captions: python -m src.cli caption --video-path path/to/video.mp4"
echo ""
echo "For more information, see README.md"
