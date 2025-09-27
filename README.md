# Open-source LLM Fine-tuning using PEFT/LoRA

A practical implementation of **Parameter-Efficient Fine-Tuning (PEFT)** with **Low-Rank Adaptation (LoRA)** for financial news sentiment classification using open-source Large Language Models.

## 📋 Project Overview

This repository demonstrates how to fine-tune the **Gemma-1B** model for financial news sentiment classification. The project compares the performance of a base pre-trained model against a fine-tuned version using the PEFT/LoRA technique, providing insights into the effectiveness of parameter-efficient fine-tuning approaches.

### Problem Statement

Classify financial news articles into three sentiment categories:
- **Positive** 📈
- **Negative** 📉  
- **Neutral** ➖

## 🚀 Methodology

The notebook explores two complementary approaches:

### 1. Zero-Shot Classification (Baseline)
- Uses the base `Gemma-1B` model without fine-tuning
- Establishes performance baseline for comparison
- Tests the model's inherent understanding of financial sentiment

### 2. Fine-Tuned Classification (PEFT/LoRA)
- Fine-tunes `Gemma-1B` using Parameter-Efficient Fine-Tuning
- Implements Low-Rank Adaptation for efficient parameter updates
- Optimized for Google Colab environment

## 📊 Dataset

**Source**: [Daniel-ML/sentiment-analysis-for-financial-news-v2](https://huggingface.co/datasets/Daniel-ML/sentiment-analysis-for-financial-news-v2)

- **Total Samples**: 4,846 financial news articles
- **Train Set**: 4,361 samples (90%)
- **Test Set**: 485 samples (10%)
- **Format**: Direct loading from Hugging Face Hub

## 🛠️ Key Dependencies

```python
# Core ML Libraries
transformers          # Pre-trained models and tokenizers
peft                  # Parameter-Efficient Fine-Tuning
torch                 # PyTorch framework

# Optimization & Training
trl                   # SFTTrainer for fine-tuning
bitsandbytes          # 4-bit quantization for memory efficiency

# Data Processing
datasets              # Dataset handling and processing
```

## 📈 Results Summary

### Base Model Performance (Zero-Shot)
- **Accuracy**: 45.98%
- **Strengths**: Good performance on 'negative' sentiment classification
- **Weaknesses**: Struggles with 'neutral' and 'positive' labels
- **Observation**: Shows inherent bias towards certain sentiment classes

### Fine-Tuned Model Performance (PEFT/LoRA)
- **Accuracy**: 41.24% (1 epoch)
- **Accuracy Change**: -4.74% compared to base model
- **Key Findings**:
  - Improved recall for 'positive' class (0.99 vs 0.94)
  - Tendency to over-classify as 'positive' sentiment
  - Highlights importance of hyperparameter tuning

## 🎯 Key Insights

1. **Unexpected Results**: Fine-tuning didn't automatically improve performance, emphasizing the critical role of:
   - Hyperparameter optimization
   - Data preprocessing quality
   - Training epoch selection

2. **Class Imbalance**: Both models struggle with 'neutral' classification, suggesting potential dataset imbalance issues

3. **Overfitting Indicators**: The fine-tuned model's bias toward 'positive' classification may indicate overfitting to training patterns

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local GPU environment
- Hugging Face account for dataset access

### Installation
```bash
pip install transformers peft torch trl bitsandbytes datasets
```

### Usage
1. Clone this repository
2. Open the Jupyter notebook in Google Colab
3. Run all cells to reproduce the experiment
4. Compare results between base and fine-tuned models

## 🔬 Future Improvements

- **Hyperparameter Tuning**: Systematic optimization of learning rate, epochs, and LoRA parameters
- **Data Augmentation**: Expand training data to address class imbalance
- **Advanced PEFT Techniques**: Experiment with other parameter-efficient methods
- **Model Ensemble**: Combine multiple fine-tuned models for improved performance

## 📝 Technical Notes

- **Memory Optimization**: Uses 4-bit quantization via bitsandbytes
- **Training Environment**: Optimized for Google Colab's resource constraints
- **Efficiency Focus**: PEFT/LoRA reduces trainable parameters while maintaining model capability

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Hyperparameter optimization experiments
- Alternative PEFT techniques implementation
- Performance analysis and visualization enhancements
- Documentation improvements

## 📄 License

This project is open-source and available under the MIT License.

---
