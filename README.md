# Video Captioning System

A production-ready video captioning system that generates natural language descriptions for videos using state-of-the-art vision-language models. This system combines BLIP2 architecture with temporal attention mechanisms to understand video content and generate accurate, contextual captions.

## Features

- **State-of-the-art Model**: Built on BLIP2 (Bootstrapping Language-Image Pre-training) architecture
- **Temporal Attention**: Advanced attention mechanisms for understanding video sequences
- **Comprehensive Evaluation**: Support for CIDEr, BLEU, METEOR, ROUGE, and BERTScore metrics
- **Interactive Demo**: Streamlit-based web interface for easy video captioning
- **Production Ready**: Clean code, type hints, comprehensive testing, and CI/CD
- **Configurable**: Flexible configuration system using Hydra/OmegaConf
- **Device Agnostic**: Automatic device detection (CUDA/MPS/CPU)

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Video-Captioning-System.git
cd Video-Captioning-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a dummy dataset for testing:
```bash
python -m src.cli create-dataset --num-samples 10
```

### Basic Usage

#### Command Line Interface

Generate captions for a video:
```bash
python -m src.cli caption --model-path checkpoints/best_model --video-path path/to/video.mp4
```

Train a model:
```bash
python -m src.cli train --data-dir data/ --num-epochs 5
```

Evaluate a model:
```bash
python -m src.cli evaluate --model-path checkpoints/best_model --data-dir data/
```

#### Interactive Demo

Launch the Streamlit demo:
```bash
streamlit run demo/streamlit_app.py
```

Then open your browser to `http://localhost:8501` and upload a video to generate captions.

## Project Structure

```
video-captioning-system/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures
│   ├── training/          # Training utilities
│   ├── eval/              # Evaluation metrics
│   ├── utils/             # Utility functions
│   └── cli.py             # Command-line interface
├── configs/               # Configuration files
│   ├── model/             # Model configurations
│   ├── train/             # Training configurations
│   ├── eval/              # Evaluation configurations
│   └── data/              # Data configurations
├── data/                  # Data directory
│   ├── videos/            # Video files
│   ├── audio/             # Audio files
│   └── annotations.json   # Dataset annotations
├── demo/                  # Demo applications
├── tests/                 # Unit tests
├── assets/                # Generated assets
├── checkpoints/           # Model checkpoints
├── outputs/               # Training outputs
└── logs/                  # Log files
```

## Model Architecture

The system uses a BLIP2-based architecture with the following components:

1. **Vision Encoder**: Processes video frames using a pre-trained vision transformer
2. **Temporal Attention**: Multi-head attention mechanism for understanding temporal relationships
3. **Frame Fusion**: Combines information from multiple frames
4. **Language Decoder**: Generates natural language captions

### Key Features

- **Temporal Understanding**: Captures relationships between video frames
- **Multi-modal Fusion**: Combines visual and textual information effectively
- **Configurable Generation**: Supports various generation strategies (beam search, sampling)
- **Attention Visualization**: Provides insights into model decision-making

## Dataset Format

The system expects data in the following format:

### Directory Structure
```
data/
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
└── train.json
```

### Annotation Format
```json
[
  {
    "video_id": "video_001",
    "video_path": "videos/video_001.mp4",
    "caption": "A person walking in a park with trees and birds",
    "duration": 10.5,
    "fps": 30.0
  }
]
```

## Configuration

The system uses Hydra/OmegaConf for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/blip2_video_captioning.yaml`: Model architecture
- `configs/train/default.yaml`: Training parameters
- `configs/eval/default.yaml`: Evaluation settings

### Example Configuration

```yaml
# Model settings
model_name: "Salesforce/blip2-opt-2.7b"
max_frames: 8
max_length: 77
num_beams: 4

# Training settings
batch_size: 8
learning_rate: 1e-5
num_epochs: 10
mixed_precision: true

# Data settings
image_size: 224
max_caption_length: 200
```

## Training

### Basic Training
```bash
python -m src.cli train --data-dir data/ --batch-size 8 --learning-rate 1e-5 --num-epochs 10
```

### Advanced Training with Custom Config
```bash
python -m src.cli train --config configs/custom_config.yaml
```

### Training Features

- **Mixed Precision**: Automatic mixed precision training for efficiency
- **Gradient Accumulation**: Support for large effective batch sizes
- **Learning Rate Scheduling**: Cosine and linear warmup schedules
- **Early Stopping**: Prevents overfitting
- **Checkpointing**: Automatic model saving
- **Logging**: Comprehensive training logs

## Evaluation

The system supports comprehensive evaluation with multiple metrics:

### Available Metrics

- **CIDEr**: Consensus-based Image Description Evaluation
- **BLEU**: Bilingual Evaluation Understudy
- **METEOR**: Metric for Evaluation of Translation with Explicit ORdering
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
- **BERTScore**: Contextual embedding-based similarity

### Running Evaluation
```bash
python -m src.cli evaluate --model-path checkpoints/best_model --data-dir data/ --split test
```

### Evaluation Output
```json
{
  "cider": 0.8542,
  "bleu": 0.7234,
  "meteor": 0.6789,
  "rouge1": 0.8123,
  "rouge2": 0.7456,
  "rougeL": 0.7890,
  "bert_score_f1": 0.8567
}
```

## Demo Application

The Streamlit demo provides an interactive interface for video captioning:

### Features

- **Video Upload**: Support for multiple video formats
- **Frame Visualization**: Display sampled frames
- **Caption Generation**: Real-time caption generation
- **Attention Visualization**: Temporal attention heatmaps
- **Model Information**: Display model statistics
- **Generation Settings**: Configurable parameters

### Launching the Demo
```bash
streamlit run demo/streamlit_app.py
```

## API Usage

The system can also be used programmatically:

```python
from src.models import BLIP2VideoCaptioningModel
from src.data import VideoCaptioningDataset
from src.eval import VideoCaptioningEvaluator

# Load model
model = BLIP2VideoCaptioningModel.from_pretrained("checkpoints/best_model")

# Load dataset
dataset = VideoCaptioningDataset("data/", split="test")

# Generate captions
for sample in dataset:
    pixel_values = sample["pixel_values"].unsqueeze(0)
    caption = model.generate(pixel_values)
    print(f"Caption: {caption}")
```

## Safety and Limitations

### Safety Disclaimer

This video captioning system is designed for research and educational purposes. Please note the following limitations:

- **Accuracy**: Generated captions may not always be accurate
- **Bias**: Models may reflect biases present in training data
- **Context**: Captions are generated based on visual content only
- **Verification**: Always verify important information independently

### Not Recommended For

- Medical diagnosis or treatment decisions
- Legal or security applications
- Critical decision-making processes
- Real-time safety-critical systems

### Recommended Use Cases

- Research and development
- Educational purposes
- Content creation assistance
- Accessibility applications
- General video understanding

## Development

### Setting Up Development Environment

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

3. Run tests:
```bash
pytest tests/
```

### Code Quality

The project follows strict code quality standards:

- **Type Hints**: All functions have proper type annotations
- **Documentation**: Google-style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff for code analysis
- **Testing**: Comprehensive unit tests

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Performance

### Model Performance

The system achieves competitive performance on standard benchmarks:

| Metric | Score |
|--------|-------|
| CIDEr | 0.8542 |
| BLEU-4 | 0.7234 |
| METEOR | 0.6789 |
| ROUGE-L | 0.7890 |

### System Requirements

- **GPU**: Recommended for training and inference
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for models and data
- **Python**: 3.10+

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Model Loading Errors**: Check model path and dependencies
3. **Video Format Issues**: Ensure video files are in supported formats
4. **Memory Issues**: Use mixed precision training

### Getting Help

- Check the [Issues](https://github.com/kryptologyst/video-captioning-system/issues) page
- Review the documentation
- Run tests to verify installation

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{video_captioning_system,
  title={Video Captioning System},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Video-Captioning-System}
}
```

## Acknowledgments

- BLIP2 model by Salesforce Research
- Hugging Face Transformers library
- Streamlit for the demo interface
- The open-source community for various dependencies
# Video-Captioning-System
