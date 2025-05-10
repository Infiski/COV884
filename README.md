# COV884

# Knowledge Graph QA Pipeline

## Overview

This Jupyter Notebook implements a complete pipeline for Knowledge Graph-based Question Answering:

1. **Extract Entities and Triplets**: Use custom processing tools to extract key entities and relations from raw text.
2. **Knowledge Graph Creation**: Build a directed graph using NetworkX with extracted triplets.
3. **Q/A Generation via KG**: Generate question-answer pairs using a HuggingFace model wrapper and LangChain prompts.
4. **Path Sampling**: Sample semantically coherent reasoning paths from the knowledge graph.
5. **Fine-Tuning**: Fine-tune a downstream model on the generated Q/A pairs using LoRA or other methods.

## Prerequisites

- Python 3.8 or higher
- A Hugging Face API token (`HF_API_KEY`) stored in a `.env` file.
- A local module `processing.py` with data processing utilities.

## Installation

1. Clone the repository or download the notebook and supporting files.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root with your Hugging Face API key:
   ```
   HF_API_KEY=your_huggingface_api_token_here
   ```

## Usage

1. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook Untitled-1.ipynb
   ```
2. Execute the cells in order. Adjust parameters such as `model_id`, `temperature`, or dataset paths as needed.
3. Inspect outputs at each pipeline stage and customize for your own data.

## File Structure

- `Untitled-1.ipynb` – The main notebook containing the pipeline.
- `processing.py` – Helper functions for entity extraction and triplet generation.
- `requirements.txt` – List of Python dependencies.
- `.env` – Environment variables (not included in version control).

## License

This project is released under the MIT License.
