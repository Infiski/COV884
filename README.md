# COV884

# Knowledge Graph QA Pipeline

## Overview

This repository implements a pipeline for generating and evaluating question-answer pairs using knowledge graph techniques. It includes scripts for generation, training, and evaluation, alongside sample output.

## File Structure

```
.
├── qa_gen.py                  # Generate question-answer pairs from knowledge graph paths
├── train.py                   # Train a model on generated Q/A pairs
├── evaluate.py                # Evaluate the trained model and compute metrics
├── formatted_output.json_scores.json  # Sample evaluation output with scores
└── requirements.txt           # List of Python dependencies
```

## Requirements

Make sure you have Python 3.8 or higher installed. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. **Generate Q/A pairs**  
   ```bash
   python qa_gen.py --input <path_to_input_data> --output generated_qa.json
   ```

2. **Train Model**  
   ```bash
   python train.py --qa_path generated_qa.json --model_output model_checkpoint/
   ```

3. **Evaluate Model**  
   ```bash
   python evaluate.py --model_checkpoint model_checkpoint/ --test_data generated_qa.json --output output.json
   ```

Adjust script arguments as needed (e.g., file paths, hyperparameters).

## License

This project is released under the MIT License.
