import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict


# Dataset Class
class TextOnlyQADataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=128):
        """
        Args:
            json_file (str): Path to the JSON file containing the dataset.
            tokenizer: Hugging Face tokenizer for text preprocessing.
            max_length (int): Maximum token length for padding/truncation.
        """
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

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

        # Tokenize question + each choice
        inputs = [
            self.tokenizer(
                f"{question} {choice}",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            for choice in choices
        ]

        return {
            "input_ids": torch.cat(
                [inp["input_ids"] for inp in inputs], dim=0
            ),  # (num_choices, seq_len)
            "attention_mask": torch.cat(
                [inp["attention_mask"] for inp in inputs], dim=0
            ),  # (num_choices, seq_len)
            "answer_idx": answer_idx,  # Correct choice index
            "category": item["question_id"].split("_")[0],  # Question category
        }


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
