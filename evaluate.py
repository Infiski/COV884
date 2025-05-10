import json
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel, PeftConfig
from difflib import SequenceMatcher

# Load test data
def load_test_dataset(jsonl_path):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            q = item["question"].strip()
            a = item["answer"].strip()
            prompt = f"<s>[INST] Answer the following question:\n\n{q}\n[/INST]\n"
            data.append({"input": prompt, "gold": a})
    return Dataset.from_list(data)

# Exact match
def exact_match(a, b):
    return a.strip().lower() == b.strip().lower()

# Similarity score
def string_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Load model
def load_lora_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def evaluate(model_dir, test_file):
    model, tokenizer = load_lora_model(model_dir)
    dataset = load_test_dataset(test_file)

    gen_config = GenerationConfig(
        max_new_tokens=50,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )

    em_total = 0
    sim_total = 0
    n = len(dataset)

    print("Evaluating...")
    for item in tqdm(dataset):
        prompt = item["input"]
        gold = item["gold"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, generation_config=gen_config)
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the answer portion (after [/INST])
        if "[/INST]" in gen_text:
            gen_ans = gen_text.split("[/INST]")[-1].strip()
        else:
            gen_ans = gen_text.strip()

        # Metrics
        em_total += exact_match(gen_ans, gold)
        sim_total += string_similarity(gen_ans, gold)

        print(f"\nQ: {prompt.strip()}\nGT: {gold}\nPred: {gen_ans}\n")

    print("\n--- Evaluation Summary ---")
    print(f"Total Samples   : {n}")
    print(f"Exact Match     : {em_total / n:.4f}")
    print(f"String Similarity (avg): {sim_total / n:.4f}")

if __name__ == "__main__":
    evaluate(
        model_dir="./mistral-qa-final",        
        test_file="qa_test_data.jsonl"         
    )
