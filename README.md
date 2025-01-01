# Advanced Sentiment Analysis Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange.svg)](https://tensorflow.org/)

## Project Overview
A sophisticated sentiment analysis system implementing state-of-the-art deep learning architectures. This project showcases the implementation and comparison of multiple neural network architectures for sentiment classification, featuring advanced preprocessing techniques and performance optimization.

## Technical Architecture
### Model Implementations
1. **Bidirectional LSTM**
   - Dual-layer architecture with attention mechanism
   - Dropout regularization for preventing overfitting
   - Advanced embedding layer with GloVe pre-trained vectors

2. **GRU (Gated Recurrent Unit)**
   - Bidirectional architecture
   - Optimized for faster training while maintaining accuracy
   - Dense layer with ReLU activation

3. **Simple RNN**
   - Baseline model for performance comparison
   - Lightweight architecture for rapid deployment

4. **Conv1D Neural Network**
   - 1D convolutional layers for feature extraction
   - Global max pooling for dimensional reduction
   - Dense layers with dropout for classification

## Performance Metrics

| Model Architecture | Accuracy | F1 Score | Training Time |
|-------------------|----------|-----------|---------------|
| Bidirectional LSTM| 0.8623   | 0.8591    | 45m 32s      |
| GRU               | 0.8591   | 0.8567    | 38m 15s      |
| Simple RNN        | 0.8456   | 0.8412    | 25m 43s      |
| Conv1D            | 0.8689   | 0.8645    | 32m 18s      |

## Performance Visualization
### Model Comparison
![Model Comparison](./sentiment_analysis_custom/model_comparison.png)

### Training Metrics
![Training Metrics](./sentiment_analysis_custom/training_metrics.png)

## Key Features
- **Advanced Preprocessing Pipeline**
  - Text normalization and tokenization
  - GloVe embeddings integration
  - Efficient batch processing

- **Model Architecture**
  - Bidirectional layers for context capture
  - Attention mechanisms for important feature focus
  - Dropout layers for regularization

- **Performance Optimization**
  - TensorFlow optimization for GPU acceleration
  - Batch size optimization for memory efficiency
  - Early stopping with model checkpointing

## Quick Start

```bash
# Clone repository
git clone https://github.com/Uzumaki-na/Sentiment-Analysis-by-RNNs.git

# Install dependencies
pip install -r requirements.txt

# Train models
python sentiment_analysis.py --train --epochs 5

# Run inference
python sentiment_app.py
```

## Technical Requirements
- Python 3.8+
- TensorFlow 2.8+
- CUDA compatible GPU (recommended)
- 8GB RAM minimum

## Project Structure
```
sentiment-analysis/
│
├── models/
│   ├── lstm.py
│   ├── gru.py
│   └── conv1d.py
│
├── utils/
│   ├── preprocessing.py
│   └── metrics.py
│
├── sentiment_analysis.py
├── sentiment_app.py
└── requirements.txt
```



