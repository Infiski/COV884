#usage python multihop_qa_generator.py --input your_input_file.jsonl --output multihop_results.json

import json
import re
import os
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MistralWrapper:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2", api_key=None, temperature=0.1, max_tokens=1024):
        """
        Initialize the Mistral model wrapper.
        
        Args:
            model_id: The Hugging Face model ID
            api_key: Hugging Face API key (if None, loads from HF_API_KEY env variable)
            temperature: Temperature for text generation
            max_tokens: Maximum number of tokens to generate
        """
        self.model_id = model_id
        self.api_key = api_key if api_key else os.getenv("HF_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided. Set HF_API_KEY environment variable or pass api_key parameter.")
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def __call__(self, prompt):
        """
        Generate a response using the Mistral model.
        
        Args:
            prompt: The input prompt
            
        Returns:
            The generated text
        """
        try:
            # Format the prompt according to Mistral's chat template
            formatted_prompt = f"""<s>[INST] {prompt} [/INST]"""
            
            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "return_full_text": False,
                    "do_sample": True,
                }
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            return response.json()[0]["generated_text"]
        except Exception as e:
            print(f"Error calling Hugging Face API: {e}")
            return None

def parse_json_document(json_line):
    """
    Parse the JSON document and extract facts and passages.
    
    Args:
        json_line: A line from the input file containing JSON data
        
    Returns:
        A tuple of (facts, passages, title)
    """
    try:
        data = json.loads(json_line)
        document = data.get("document", "")
        title = data.get("chapter_title", "")
        
        # Extract facts
        facts_match = re.search(r'<facts>(.*?)<\/facts>', document, re.DOTALL)
        facts = []
        if facts_match:
            facts_text = facts_match.group(1)
            fact_matches = re.findall(r'<fact>(.*?)<\/fact>', facts_text, re.DOTALL)
            facts = [fact.strip() for fact in fact_matches]
        
        # Extract passages
        passages_match = re.search(r'<passages>(.*?)<\/passages>', document, re.DOTALL)
        passages = []
        if passages_match:
            passages_text = passages_match.group(1)
            passage_matches = re.findall(r'<passage>(.*?)<\/passage>', passages_text, re.DOTALL)
            passages = [passage.strip() for passage in passage_matches]
        
        return facts, passages, title
    except json.JSONDecodeError:
        print(f"Error parsing JSON line: {json_line[:100]}...")
        return [], [], ""

def create_multihop_prompt(facts, passages):
    """
    Create a prompt for generating multi-hop questions based on facts and passages.
    
    Args:
        facts: List of facts
        passages: List of passages
        
    Returns:
        A formatted prompt string
    """
    facts_text = "\n".join([f"<fact>{fact}</fact>" for fact in facts])
    passages_text = "\n".join([f"<passage>{passage}</passage>" for passage in passages])
    
    prompt = f"""You are a Multi-Hop Factual Question Formulation Assistant. Generate precise, fact-based, multi-hop questions requiring integration of specific information from provided texts, resulting in very brief factoid answers (2-3 words maximum).

Given the following facts and passages, please generate multi-hop questions:

Facts:
{facts_text}

Passages:
{passages_text}

Generate 1-2 multi-hop questions that require integrating multiple pieces of information from these texts.
Each question should have a clear factoid answer (2-3 words maximum).
Begin with subquestions that build towards a final complex question.
Please follow this exact format:

<sub-question>[First focused question]</sub-question>
<answer-to-sub-question>[Concise answer]</answer-to-sub-question>

<sub-question>[Follow-up question using previous answer]</sub-question>
<answer-to-sub-question>[Concise answer]</answer-to-sub-question>

[Additional sub-questions as needed]

<question-type>[Number of reasoning steps, e.g., "2-hop"]</question-type>

<complex-question>[Question that integrates all previous sub-questions]</complex-question>
<answer-to-complex-question>[Final answer, 2-3 words]</answer-to-complex-question>

<explanation>[Break down how to answer the complex question step-by-step]</explanation>
<done>
"""
    return prompt

def extract_qa_from_response(response):
    """
    Extract the multi-hop questions, answers, and explanations from the model's response.
    
    Args:
        response: The model's generated text
        
    Returns:
        A dictionary containing the extracted QA components
    """
    if not response:
        return None
    
    # Extract components using regex
    qa_dict = {
        "sub_questions": [],
        "sub_answers": [],
        "question_type": None,
        "complex_question": None,
        "complex_answer": None,
        "explanation": None
    }
    
    # Extract sub-questions and their answers
    sub_questions = re.findall(r'<sub-question>(.*?)<\/sub-question>', response, re.DOTALL)
    sub_answers = re.findall(r'<answer-to-sub-question>(.*?)<\/answer-to-sub-question>', response, re.DOTALL)
    
    for i in range(min(len(sub_questions), len(sub_answers))):
        qa_dict["sub_questions"].append(sub_questions[i].strip())
        qa_dict["sub_answers"].append(sub_answers[i].strip())
    
    # Extract question type
    question_type_match = re.search(r'<question-type>(.*?)<\/question-type>', response, re.DOTALL)
    if question_type_match:
        qa_dict["question_type"] = question_type_match.group(1).strip()
    
    # Extract complex question and answer
    complex_q_match = re.search(r'<complex-question>(.*?)<\/complex-question>', response, re.DOTALL)
    if complex_q_match:
        qa_dict["complex_question"] = complex_q_match.group(1).strip()
    
    complex_a_match = re.search(r'<answer-to-complex-question>(.*?)<\/answer-to-complex-question>', response, re.DOTALL)
    if complex_a_match:
        qa_dict["complex_answer"] = complex_a_match.group(1).strip()
    
    # Extract explanation
    explanation_match = re.search(r'<explanation>(.*?)<\/explanation>', response, re.DOTALL)
    if explanation_match:
        qa_dict["explanation"] = explanation_match.group(1).strip()
    
    return qa_dict

def process_jsonl_file(input_file, output_file, model):
    """
    Process a JSONL file containing documents and generate multi-hop questions.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to save the output JSON results
        model: The language model wrapper
        
    Returns:
        A list of results containing generated multi-hop QAs
    """
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line_idx, line in enumerate(tqdm(lines, desc="Processing documents")):
        try:
            facts, passages, title = parse_json_document(line)
            
            if not facts or not passages:
                print(f"Skipping document {line_idx} - missing facts or passages")
                continue
            
            prompt = create_multihop_prompt(facts, passages)
            response = model(prompt)
            
            qa_data = extract_qa_from_response(response)
            
            if qa_data:
                result = {
                    "title": title,
                    "facts": facts,
                    "passages": passages,
                    "qa_data": qa_data
                }
                results.append(result)
                print(f"Generated QA for document {line_idx} - {title}")
            else:
                print(f"Failed to extract QA from response for document {line_idx}")
        
        except Exception as e:
            print(f"Error processing document {line_idx}: {e}")
    
    # Save the results to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

def process_text_data(text_data, output_file=None):
    """
    Process a string containing multiple JSON documents and generate multi-hop questions.
    
    Args:
        text_data: String containing JSON documents (one per line)
        output_file: Optional path to save the output JSON results
        
    Returns:
        A list of results containing generated multi-hop QAs
    """
    # Initialize the Mistral model
    model = MistralWrapper(model_id="mistralai/Mixtral-8x22B-Instruct-v0.1")
    
    results = []
    lines = text_data.strip().split('\n')
    
    for line_idx, line in enumerate(tqdm(lines, desc="Processing documents")):
        try:
            facts, passages, title = parse_json_document(line)
            
            if not facts or not passages:
                print(f"Skipping document {line_idx} - missing facts or passages")
                continue
            
            prompt = create_multihop_prompt(facts, passages)
            response = model(prompt)
            
            qa_data = extract_qa_from_response(response)
            
            if qa_data:
                result = {
                    "title": title,
                    "facts": facts,
                    "passages": passages,
                    "qa_data": qa_data
                }
                results.append(result)
                print(f"Generated QA for document {line_idx} - {title}")
            else:
                print(f"Failed to extract QA from response for document {line_idx}")
        
        except Exception as e:
            print(f"Error processing document {line_idx}: {e}")
    
    # Save the results to the output file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate multi-hop QA pairs from documents')
    parser.add_argument('--input', type=str, help='Input JSONL file path')
    parser.add_argument('--output', type=str, default='multihop_qa_results.json', help='Output JSON file path')
    parser.add_argument('--model', type=str, default='mistralai/Mixtral-8x22B-Instruct-v0.1', help='Hugging Face model ID')
    
    args = parser.parse_args()
    
    # Initialize the language model
    try:
        model = MistralWrapper(model_id=args.model)
        print(f"Using model: {args.model}")
    except ValueError as e:
        print(f"Error initializing model: {e}")
        return
    
    # Process the input file
    results = process_jsonl_file(args.input, args.output, model)
    print(f"Processed {len(results)} documents. Results saved to {args.output}.")

if __name__ == "__main__":
    main()