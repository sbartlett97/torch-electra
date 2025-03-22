# ELECTRA Training Implementation

A PyTorch-based implementation of the ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately) pre-training method using the HuggingFace Transformers library. This implementation focuses on providing an easy-to-use and extensible framework for pre-training transformer models using the ELECTRA approach.

## Features

- 🚀 Easy-to-use training pipeline
- 🔧 Support for custom model configurations
- 📊 Training progress tracking with loss curves
- ⚡ Mixed precision training support
- 🔄 Gradient accumulation for handling large batch sizes
- 🎛️ Hyperparameter optimization using Optuna
- 💾 Automatic checkpointing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/electra-implementation.git
cd electra-implementation

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Train a model using default settings (base ELECTRA configuration):
```bash
python main.py --run_name my_electra_model
```

### Training Options

Choose from different model sizes:
```bash
# Small ELECTRA
python main.py --preset small --run_name electra_small

# Base ELECTRA (default)
python main.py --preset base --run_name electra_base

# Large ELECTRA
python main.py --preset large --run_name electra_large
```

Customize training parameters:
```bash
python main.py \
    --preset base \
    --batch_size 32 \
    --steps 1000000 \
    --dataset_path "your/dataset/path" \
    --run_name custom_electra
```

Run hyperparameter optimization:
```bash
python main.py --preset base --optuna
```

## Model Architecture

The implementation follows the original ELECTRA paper's architecture:

- **Generator**: Smaller transformer model that predicts masked tokens
- **Discriminator**: Larger transformer model that learns to distinguish between real and replaced tokens
- **Shared embeddings** between generator and discriminator
- **Size ratios**:
  - Small: 12-layer discriminator, 12-layer generator (generator has 1/3 the width)
  - Base: 12-layer discriminator, 12-layer generator (generator has 1/3 the width)
  - Large: 24-layer discriminator, 24-layer generator (generator has 1/4 the width)

## Training Details

- Uses masked language modeling (MLM) for the generator
- Implements replaced token detection (RTD) for the discriminator
- Supports gradient accumulation for effective batch sizes
- Implements learning rate warmup and decay
- Uses mixed precision training for improved performance
- Tracks and saves training metrics

## Results and Metrics

Training progress can be monitored through:
- Real-time loss tracking in the console
- Generated loss curves (saved as `loss_curve.png`)
- Training logs (saved as `training_log.csv`)

## References

This implementation is based on the original ELECTRA paper and inspired by existing implementations:

- **ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators**
  Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning  
  [ICLR 2020](https://openreview.net/pdf?id=r1xMH1BtvB)  
  ```bibtex
  @inproceedings{clark2020electra,
    title = {{ELECTRA}: Pre-training Text Encoders as Discriminators Rather Than Generators},
    author = {Kevin Clark and Minh-Thang Luong and Quoc V. Le and Christopher D. Manning},
    booktitle = {ICLR},
    year = {2020},
    url = {https://openreview.net/pdf?id=r1xMH1BtvB}
  }
  ```

- **PyTorch Implementation of ELECTRA**
  Richard Wang  
  [GitHub Repository](https://github.com/richarddwang/electra_pytorch)  
  ```bibtex
  @misc{electra_pytorch,
    author = {Richard Wang},
    title = {PyTorch implementation of ELECTRA},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/richarddwang/electra_pytorch}}
  }
  ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- HuggingFace team for their Transformers library
- PyTorch team and community
- Original ELECTRA paper authors
