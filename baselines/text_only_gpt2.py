import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from baselines.text_only_dataset import TextOnlyQADataset


# Evaluation Function
def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    total = 0
    correct = 0

    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
            input_ids = batch["input_ids"].to(
                device
            )  # Shape: (batch_size * num_choices, seq_len)
            attention_mask = batch["attention_mask"].to(
                device
            )  # Shape: (batch_size * num_choices, seq_len)
            answer_idx = batch["answer_idx"].to(device)  # Shape: (batch_size,)

            batch_size, num_choices, seq_len = (
                input_ids.size(0),
                input_ids.size(1),
                input_ids.size(2),
            )

            log_probs = []
            for i in range(num_choices):
                # Get logits from the model
                outputs = model(
                    input_ids=input_ids[:, i], attention_mask=attention_mask[:, i]
                )
                logits = outputs.logits  # Shape: (seq_len, vocab_size)

                # Compute log probabilities of the last token (choice end)
                last_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
                last_token_probs = F.log_softmax(last_token_logits, dim=-1)

                # Compute log probability of the actual choice text
                choice_tokens = input_ids[:, i][:, -1]  # Last token of each choice
                log_prob_sum = torch.gather(
                    last_token_probs, index=choice_tokens.unsqueeze(-1), dim=-1
                ).squeeze(-1)

                log_probs.append(log_prob_sum)

            # Stack log probabilities and predict the best choice
            log_probs = torch.stack(
                log_probs, dim=1
            )  # Shape: (batch_size, num_choices)
            predictions = torch.argmax(log_probs, dim=1)  # Predicted choice index

            correct += (predictions == answer_idx).sum().item()
            total += len(answer_idx)

            # Update category-wise counts
            for i in range(len(batch["category"])):
                category_correct[batch["category"][i]] += int(
                    predictions[i] == answer_idx[i]
                )
                category_total[batch["category"][i]] += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

    # Compute category-wise accuracy
    category_accuracies = {}
    for category, total in category_total.items():
        category_accuracies[category] = category_correct[category] / total
        print(f"{category} Accuracy: {category_accuracies[category]:.4f}")

    return accuracy, category_accuracies


# Main Script
if __name__ == "__main__":
    # Paths and Parameters
    json_file_path = "/data/user_data/gdhanuka/STAR_dataset/STAR_val.json"  # Replace with actual path to your dataset JSON file
    model_name = "openai-community/gpt2"  # You can replace this with a smaller LLaMA model if available
    batch_size = 32
    max_length = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Tokenizer and Dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = TextOnlyQADataset(
        json_file=json_file_path, tokenizer=tokenizer, max_length=max_length
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Load Model and Move to Device
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Evaluate Model on Dataset
    evaluate(model=model, dataloader=dataloader, tokenizer=tokenizer, device=device)
