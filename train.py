import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

def load_qa_data(jsonl_path):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            q = item["question"].strip()
            a = item["answer"].strip()
            full_text = f"<s>[INST] Answer the following question:\n\n{q}\n[/INST]\n{a}</s>"
            data.append({"text": full_text})
    return Dataset.from_list(data)

def tokenize_fn(examples, tokenizer, max_length=2048):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

def main():
    model_id = "mistralai/Mistral-7B-v0.1"
    jsonl_path = "qa_data.jsonl"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # Load and tokenize dataset
    dataset = load_qa_data(jsonl_path)
    dataset = dataset.train_test_split(test_size=0.05)
    tokenized = dataset.map(lambda x: tokenize_fn(x, tokenizer), batched=True)
    tokenized = tokenized.remove_columns(["text"])

    # Training setup
    training_args = TrainingArguments(
        output_dir="./checkpoints-mistral-qa",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=4,
        fp16=True,
        logging_dir="./logs",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    # Train & Save
    trainer.train()
    model.save_pretrained("./mistral-qa-final")
    tokenizer.save_pretrained("./mistral-qa-final")

if __name__ == "__main__":
    main()
