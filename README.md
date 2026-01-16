# Uyghur Automatic Speech Recognition (ASR)

A deep learning-based speech recognition system for the Uyghur language using OpenAI's Whisper model.

## Overview

This project implements an end-to-end ASR system for transcribing Uyghur speech. It fine-tunes the Whisper model on a custom Uyghur speech dataset containing over 23 hours of audio.

### Key Features

- **Whisper-based Architecture**: Leverages OpenAI's state-of-the-art Whisper model
- **Modular Design**: Clean, maintainable code structure with separate modules
- **GPU Optimized**: Mixed precision training with gradient checkpointing
- **Low CER**: Optimized for Character Error Rate metric

## Project Structure

```
├── config.py           # Configuration settings
├── data_loader.py      # Dataset loading utilities
├── preprocessor.py     # Audio preprocessing
├── data_collator.py    # Batch collation for training
├── model.py            # Whisper model initialization
├── metrics.py          # CER evaluation metrics
├── trainer.py          # Training pipeline
├── inference.py        # Prediction generation
├── utils.py            # Utility functions
├── main.py             # Entry point script
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Requirements

- Python
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- PyTorch

## Installation

```bash
# Clone the repository
git clone https://github.com/vishwas-mehta/Speech-To-Text.git
cd Speech-To-Text

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The dataset should be organized as follows:

```
data/
├── wavs/           # Audio files (.wav, 16kHz, mono)
├── train.csv       # Training metadata (ID, filepath, transcription)
└── test.csv        # Test metadata (ID, filepath)
```

### Dataset Statistics

- **Total Audio**: ~23.95 hours
- **Training Samples**: 7,574
- **Test Samples**: 1,894
- **Sample Rate**: 16,000 Hz
- **Format**: WAV (mono)

## Usage

### Training

```bash
python main.py
```

The script will:
1. Load and preprocess the dataset
2. Fine-tune Whisper-medium on Uyghur speech
3. Generate predictions on the test set
4. Save the submission file and trained model

### Configuration

Modify `config.py` to adjust:

- **Model**: Change `ModelConfig.name` for different Whisper sizes
- **Training**: Adjust batch size, learning rate, and other hyperparameters
- **Paths**: Configure data and output directories

## Model Architecture

This project uses **Whisper-medium** (769M parameters) by default:

- Encoder: 24 layers, 1024 hidden size
- Decoder: 24 layers, 1024 hidden size
- Multilingual support with Uyghur adaptation

## Training Details

| Parameter | Value |
|-----------|-------|
| Batch Size | 4 (effective: 16) |
| Learning Rate | 5e-6 |
| Scheduler | Cosine |
| Max Steps | 1000 |
| Precision | FP16 |

## Evaluation

The model is evaluated using **Character Error Rate (CER)**:

```
CER = (S + D + I) / N × 100%
```

Where S=substitutions, D=deletions, I=insertions, N=total characters.

## Results

After training, outputs are saved to `whisper_results/`:

- `submission.csv`: Predictions for test set
- `final_model/`: Fine-tuned model weights

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Update tests for new features

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for the Transformers library
- NPPE-2 Challenge organizers for the dataset
