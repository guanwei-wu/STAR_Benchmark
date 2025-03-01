import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import re


# Define Dataset Class for LLM QA
class LLMQADataset(Dataset):
    def __init__(self, json_file, prompt_template_file):
        """
        Args:
            json_file (str): Path to the JSON file containing the dataset.
            prompt_template_file (str): Path to the prompt template file.
        """
        json_file = os.path.expanduser(json_file)
        with open(json_file, "r") as f:
            self.data = json.load(f)
            
        prompt_template_file = os.path.expanduser(prompt_template_file)
        with open(prompt_template_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read().strip()
            
        # Define the mapping from choice index to option letter
        self.idx_to_option = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        self.option_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        # Define prompt template
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        choices = [choice["choice"] for choice in item["choices"]]
        answer = item["answer"]
        answer_idx = next(
            i for i, choice in enumerate(item["choices"]) if choice["choice"] == answer
        )
        
        # Format the prompt using the template
        prompt = self.prompt_template.format(
            question=question,
            option_a=choices[0],
            option_b=choices[1],
            option_c=choices[2],
            option_d=choices[3]
        )
        
        return {
            "prompt": prompt,
            "answer_idx": answer_idx,
            "answer_option": self.idx_to_option[answer_idx],
            "category": item["question_id"].split("_")[0],  # Question category
            "question_id": item["question_id"],
        }


# Define Model Class
class LLMQAModel:
    def __init__(self, tokenizer, model, device="cuda"):
        """
        Args:
            tokenizer: The pre-loaded tokenizer
            model: The pre-loaded model
            device: Device to run the model on
        """
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        
        # Ensure the model is on the correct device
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        
        # Option parsing pattern - look for A, B, C, or D
        self.option_pattern = re.compile(r'(?:^|\s)([A-D])(?:\s|$|\.)')

    def generate_answers_batch(self, prompts, max_new_tokens=3):
        """
        Generate answers for a batch of prompts.
        
        Args:
            prompts: List of formatted question prompts
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            List of predicted option letters (A, B, C, or D)
        """
        # Tokenize all prompts in batch
        input_texts = list(prompts)
        inputs = self.tokenizer(input_texts, padding=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=0.1,  # Low temperature for more deterministic outputs
                top_p=0.95,
                do_sample=True,
            )
        
        # Process each output in the batch
        predicted_options = []
        for i, output in enumerate(outputs):
            full_output = self.tokenizer.decode(output, skip_special_tokens=True)
            prompt_len = len(input_texts[i])
            generated_text = full_output[prompt_len:]
            
            # Parse the output to extract the option
            option_match = self.option_pattern.search(generated_text)
            if option_match:
                predicted_options.append(option_match.group(1))
            else:
                # If no option found, try to find it in the full output
                option_match = self.option_pattern.search(full_output)
                if option_match:
                    predicted_options.append(option_match.group(1))
                else:
                    # If still no match, check for common words that might indicate the option
                    lower_text = generated_text.lower()
                    if "a" in lower_text and "option a" in lower_text:
                        predicted_options.append('A')
                    elif "b" in lower_text and "option b" in lower_text:
                        predicted_options.append('B')
                    elif "c" in lower_text and "option c" in lower_text:
                        predicted_options.append('C')
                    elif "d" in lower_text and "option d" in lower_text:
                        predicted_options.append('D')
                    else:
                        # Default to first option if no match found
                        if len(self.option_pattern.findall(generated_text)) == 0:
                            print(f"Warning: Could not parse option from output: {generated_text}")
                        predicted_options.append('A')
        
        return predicted_options
        
    def generate_answer(self, prompt, max_new_tokens=3):
        """
        Generate an answer for a single prompt (for backward compatibility).
        
        Args:
            prompt: The formatted question prompt
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            The predicted option letter (A, B, C, or D)
        """
        return self.generate_answers_batch([prompt], max_new_tokens)[0]


# Load model and tokenizer function
def load_model_and_tokenizer(model_name, device="cuda"):
    """
    Load the model and tokenizer from Hugging Face with optimizations.
    
    Args:
        model_name: Name of the model to load from Hugging Face
        device: Device to load the model on
        
    Returns:
        Tuple of (tokenizer, model)
    """
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",  # Automatically distribute model across GPUs if available
    )
    
    # Apply model optimizations if using CUDA
    if device == "cuda":
        # Enable flash attention if available (for faster attention computation)
        if hasattr(model.config, "use_flash_attention") and torch.cuda.is_available():
            model.config.use_flash_attention = True
            print("Flash attention enabled")
        
        # Apply torch.compile if using PyTorch 2.0+ for faster inference
        if hasattr(torch, "compile") and torch.__version__ >= "2.0.0":
            try:
                model = torch.compile(model)
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"Could not compile model: {e}")
    
    return tokenizer, model


# Evaluation Function
def evaluate(model, dataloader):
    """
    Evaluate the LLM QA model on a dataset with batch processing.
    
    Args:
        model: The LLM QA model
        dataloader: DataLoader containing the evaluation dataset
        
    Returns:
        Tuple of (accuracy, category_accuracies)
    """
    correct = 0
    total = 0
    
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    
    option_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    # Set up progress tracking
    progress_bar = tqdm(total=len(dataloader.dataset), desc="Evaluating")
    
    # Initialize results tracking for saving intermediate results
    results_log = []
    
    for batch_idx, batch in enumerate(dataloader):
        prompts = batch["prompt"]
        answer_idxs = batch["answer_idx"]
        answer_options = batch["answer_option"]
        question_ids = batch["question_id"]
        categories = batch["category"]
        
        print("prompts:", prompts[0])
        print("answer_options:", answer_options[0])
        print("answer_options:", answer_options[0])
        # print("prompts:", prompts[0])
        # print("prompts:", prompts[0]) 
         
        # Generate predictions for the whole batch
        predicted_options = model.generate_answers_batch(prompts)
        
        # Process results
        for i, predicted_option in enumerate(predicted_options):
            answer_option = answer_options[i]
            
            # Check if prediction is correct
            is_correct = predicted_option == answer_option
            correct += is_correct
            total += 1
            
            # Update category-wise counts
            category = categories[i]
            category_correct[category] += int(is_correct)
            category_total[category] += 1
            
            # Create result entry
            result_entry = {
                "question_id": question_ids[i],
                "predicted": predicted_option,
                "actual": answer_option,
                "correct": is_correct,
                "category": category
            }
            results_log.append(result_entry)
            
            # Only print every 20th example to reduce output spam
            if total % 20 == 0:
                print(f"Question ID: {question_ids[i]}")
                print(f"Predicted: {predicted_option}, Actual: {answer_option}")
                print(f"Correct: {is_correct}")
                print("-" * 50)
        
        # Update progress bar
        progress_bar.update(len(prompts))
        
        # Save intermediate results every 50 batches
        if batch_idx % 50 == 0 and batch_idx > 0:
            intermediate_accuracy = correct / total
            print(f"\nIntermediate Accuracy ({total} examples): {intermediate_accuracy:.4f}")
            
            # Save intermediate results
            # with open(f"llm_qa_intermediate_results_{batch_idx}.json", "w") as f:
            #     json.dump(results_log, f, indent=4)
    
    progress_bar.close()
    
    # Compute overall accuracy
    accuracy = correct / total
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Compute category-wise accuracy
    category_accuracies = {}
    print("\nCategory-wise Accuracies:")
    for category in category_total:
        category_accuracies[category] = category_correct[category] / category_total[category]
        print(f"{category}: {category_accuracies[category]:.4f} ({category_correct[category]}/{category_total[category]})")
    
    # Save detailed results
    with open("llm_qa_detailed_results.json", "w") as f:
        json.dump(results_log, f, indent=4)
    
    return accuracy, category_accuracies


def main():
    # Set up command line argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate LLM QA model on multiple-choice questions')
    
    # Add arguments
    parser.add_argument('--val_json', type=str, default="~/STAR/data/STAR_val.json",
                        help='Path to validation JSON file')
    parser.add_argument('--prompt_template', type=str, default='~/STAR/baselines/prompt.txt',
                        help='Path to prompt template file')
    parser.add_argument('--model', type=str, default="google/gemma-2-2b-it",
                        help='Hugging Face model name to use')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup parameters from arguments
    val_json = args.val_json
    prompt_template_file = args.prompt_template
    model_name = args.model
    batch_size = args.batch_size
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    print(f"Using model: {model_name}")
    print(f"Batch size: {batch_size}")
    
    # Set up performance optimizations
    torch.backends.cudnn.benchmark = True  # Enable cudnn benchmark for faster training
    
    # Load model and tokenizer with optimizations
    tokenizer, model = load_model_and_tokenizer(model_name, device)
    
    # Load Dataset and DataLoader
    print("Loading dataset...")
    val_dataset = LLMQADataset(json_file=val_json, prompt_template_file=prompt_template_file)
    print(f"Dataset size: {len(val_dataset)} examples")
    
    # Using the full validation dataset
    # Uncomment the line below if you want to test with a smaller subset
    # val_dataset = torch.utils.data.Subset(val_dataset, range(10))
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=1,  # Increase if you have multiple CPUs
        pin_memory=True  # Speed up data transfer to GPU
    )
    
    # Initialize LLM QA Model
    llm_qa_model = LLMQAModel(tokenizer=tokenizer, model=model, device=device)
    
    # Create directory for results
    import datetime
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_base = os.path.basename(model_name).replace("/", "_")
    prompt_base = os.path.splitext(os.path.basename(prompt_template_file))[0]
    results_dir = f"{model_base}_{prompt_base}_results"
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate
    print("Starting evaluation...")
    start_time = datetime.datetime.now()
    accuracy, category_accuracies = evaluate(model=llm_qa_model, dataloader=val_dataloader)
    end_time = datetime.datetime.now()
    
    # Calculate evaluation time
    eval_time = (end_time - start_time).total_seconds()
    examples_per_second = len(val_dataset) / eval_time
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Total evaluation time: {eval_time:.2f} seconds")
    print(f"Average speed: {examples_per_second:.2f} examples/second")
    
    print("\nCategory-wise Accuracies:")
    for category, acc in category_accuracies.items():
        print(f"{category}: {acc:.4f}")
    
    # Save results to a JSON file
    results = {
        "model": model_name,
        "batch_size": batch_size,
        "device": device,
        "overall_accuracy": accuracy,
        "category_accuracies": category_accuracies,
        "eval_time_seconds": eval_time,
        "examples_per_second": examples_per_second,
        "dataset_size": len(val_dataset)
    }
    
    results_file = os.path.join(results_dir, "llm_qa_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
    
# python text_only_llm_qa.py --model google/gemma-2-2b-it --batch_size 8 --val_json ~/STAR/data/STAR_val.json --prompt_template ~/STAR/baselines/prompt.txt
# python text_only_llm_qa.py --model google/gemma-2-2b-it --batch_size 8 --val_json ~/STAR/data/STAR_val.json --prompt_template ~/STAR/baselines/prompt_3_shot.txt
# python text_only_llm_qa.py --model google/gemma-2-2b-it --batch_size 8 --val_json ~/STAR/data/STAR_val.json --prompt_template ~/STAR/baselines/prompt_10_shot.txt