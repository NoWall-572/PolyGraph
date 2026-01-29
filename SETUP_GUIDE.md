# ASTRA Setup and Quick Start Guide

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ASTRA_Release

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

Download the Qwen 8B base model:
- Option 1: HuggingFace - `Qwen/Qwen3-8B`
- Option 2: ModelScope - `qwen/Qwen3-8B`
- Place in `models/Qwen3-8B/qwen/Qwen3-8B/` directory

### 3. Test with Example Data

```bash
# Parse example data
python scripts/parse_dataset.py \
    --input examples \
    --output processed_graphs/examples \
    --save

# This will create graph JSON files from the sample data
```

### 4. Run Full Pipeline

See `README.md` for detailed instructions on:
- Data generation
- Graph parsing
- Model training
- LLM fine-tuning
- Evaluation

## File Structure Overview

- **astra/generation/**: Data generation scripts
- **astra/parsing/**: Graph parsing (DHCG parser)
- **astra/data/**: Data conversion and adapter
- **astra/model/**: GNN model architecture
- **astra/training/**: Training scripts
- **astra/evaluation/**: Evaluation scripts
- **scripts/**: Utility scripts for dataset processing

## Common Issues

### Import Errors

If you encounter import errors, ensure you're running from the `ASTRA_Release` directory:
```bash
cd ASTRA_Release
python -m astra.training.train_gnn ...
```

### GPU Memory

If you run out of GPU memory:
- Use `--device cpu` for CPU-only mode
- Reduce batch size in training scripts
- Use 4-bit quantization for LLM models

### Path Issues

All paths in scripts are relative to the `ASTRA_Release` directory. Make sure you're running commands from there.

## Next Steps

1. Read `README.md` for detailed usage
2. Check `RELEASE_NOTES.md` for migration guide
3. Explore `examples/` directory for sample data
4. Run the example pipeline to verify installation

