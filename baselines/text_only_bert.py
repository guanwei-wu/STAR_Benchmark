import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
import pickle
from baselines.text_only_dataset import TextOnlyQADataset
import warnings

warnings.filterwarnings("ignore")

# Define Model Class
class MultipleChoiceQAModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(MultipleChoiceQAModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        # freeze the text encoder
        for param in self.model.parameters():
            param.requires_grad = False
        self.classifier1 = nn.Linear(self.model.config.hidden_size, 32)
        self.classifier2 = nn.Linear(32, 1)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Tensor of shape (batch_size * num_choices, seq_len)
            attention_mask: Tensor of shape (batch_size * num_choices, seq_len)

        Returns:
            scores: Tensor of shape (batch_size, num_choices)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        scores = self.classifier2(
            self.classifier1(cls_embeddings)
        )  # Shape: (batch_size * num_choices, 1)
        return scores.view(-1, 4)  # Reshape to (batch_size, num_choices)


# Training Function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", total=len(dataloader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answer_idx = batch["answer_idx"].to(device)

        optimizer.zero_grad()

        scores = model(
            input_ids=input_ids.view(-1, input_ids.size(-1)),
            attention_mask=attention_mask.view(-1, attention_mask.size(-1)),
        )  # Shape: (batch_size, num_choices)

        loss = criterion(scores, answer_idx)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss


# Evaluation Function
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            input_ids = batch["input_ids"].to(
                device
            )  # Shape: (batch_size * num_choices, seq_len)
            attention_mask = batch["attention_mask"].to(
                device
            )  # Shape: (batch_size * num_choices, seq_len)
            answer_idx = batch["answer_idx"].to(device)  # Shape: (batch_size,)

            scores = model(
                input_ids=input_ids.view(-1, input_ids.size(-1)),
                attention_mask=attention_mask.view(-1, attention_mask.size(-1)),
            )  # Shape: (batch_size, num_choices)

            predictions = torch.argmax(scores, dim=1)  # Predicted choice index
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
    train_json = "/data/user_data/gdhanuka/STAR_dataset/STAR_train.json"  # Replace with actual path to your dataset JSON file
    val_json = "/data/user_data/gdhanuka/STAR_dataset/STAR_val.json"  # Replace with actual path to your dataset JSON file
    model_name = "bert-base-uncased"
    batch_size = 16
    max_length = 128
    num_epochs = 50  # Number of epochs to train the model
    learning_rate = 1e-2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Tokenizer and Dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = TextOnlyQADataset(
        json_file=train_json, tokenizer=tokenizer, max_length=max_length
    )
    # # only keep half of the dataset (randomly chosen)
    # fraction = 2
    # train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset))[:len(train_dataset)//fraction])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_dataset = TextOnlyQADataset(
        json_file=val_json, tokenizer=tokenizer, max_length=max_length
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    # Initialize Model and Move to Device
    model = MultipleChoiceQAModel(model_name=model_name).to(device)
    optimizer = torch.optim.Adagrad(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train and Validate Model on Each Epoch
        train_loss = train(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_accuracy, _ = evaluate(
            model=model, dataloader=val_dataloader, device=device
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"New Best Validation Accuracy: {best_val_accuracy:.4f}")
