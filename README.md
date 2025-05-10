# COV884

# Knowledge Graph-Based QA Generation and Fine-Tuning Pipeline

This repository contains a Python notebook that implements a complete pipeline for building a question-answering system using Knowledge Graphs (KGs). The pipeline includes entity extraction, triplet extraction, knowledge graph construction, path sampling, and fine-tuning a language model with generated QA pairs.

## 📌 Features

- **Entity & Triplet Extraction** using LLM prompts (NER and OpenIE-style)
- **Knowledge Graph Construction** with metadata
- **Path Sampling** for multi-hop reasoning
- **Question-Answer Generation** using structured facts
- **LoRA-based Fine-Tuning** of LLaMA/Mistral models on QA data

## 🧱 Pipeline Structure

1. **Entity Extraction**  
   Uses prompt-engineered LLM calls to extract named entities from unstructured text.

2. **Triplet Extraction**  
   Extracts subject-relation-object triples (facts) from text using an OpenIE-style prompt.

3. **KG Construction**  
   Constructs a `networkx` graph using the extracted facts and maps entities to passages.

4. **Path Sampling**  
   Performs meaningful multi-hop path sampling with constraints on passage overlap and length.

5. **QA Pair Generation**  
   Generates high-quality QA pairs (with CoT explanations and paraphrases) from sampled paths.

6. **Model Fine-Tuning**  
   Fine-tunes a transformer-based LLM using LoRA on the generated QA pairs.

## 🧰 Requirements

- Python 3.8+
- `transformers`, `langchain`, `networkx`, `tqdm`, `requests`, `python-dotenv`
- Hugging Face API token in a `.env` file:
  ```env
  HF_API_KEY=your_huggingface_api_key
  ```

## 🚀 Usage

1. Clone the repo and set up the environment:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your input paragraphs and ensure `.env` is configured.

3. Run the notebook step-by-step or export the code to `.py` for modular execution.

4. The final QA dataset will be saved as JSONL and used to fine-tune your model.

## 📂 File Structure

```
.
├── Untitled-1.ipynb        # Main notebook with all scripts
├── .env                    # HuggingFace API key
├── data/                   # Input and generated data (facts, passages, questions)
├── models/                 # Fine-tuned models (optional)
└── README.md               # This file
```

## 📈 Output

- `facts.json`: Structured knowledge triples
- `qa_pairs.jsonl`: Generated QA pairs with explanations
- `fine_tuned_model/`: Saved model checkpoints

