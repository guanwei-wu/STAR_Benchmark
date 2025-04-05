import json
import os
import torch
from torch.utils.data import Dataset

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



# Define Dataset Class for further use
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
        with open(prompt_template_file, "r", encoding="utf-8") as f:
            prompt_template = f.read().strip()

        # Define the mapping from choice index to option letter
        self.idx_to_option = {0: "A", 1: "B", 2: "C", 3: "D"}
        self.option_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}

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
            option_d=choices[3],
        )

        return {
            "prompt": prompt,
            "answer_idx": answer_idx,
            "answer_option": self.idx_to_option[answer_idx],
            "category": item["question_id"].split("_")[0],  # Question category
            "question_id": item["question_id"],
        }