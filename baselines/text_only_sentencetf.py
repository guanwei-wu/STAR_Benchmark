import json
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict


# Define Dataset Class
class TextOnlyQADataset(Dataset):
    def __init__(self, json_file, max_length=128):
        """
        Args:
            json_file (str): Path to the JSON file containing the dataset.
            model: SentenceTransformer model for embedding computation.
            max_length (int): Maximum length for truncation (not used here but can be applied if needed).
        """
        with open(json_file, "r") as f:
            self.data = json.load(f)

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

        # Compute embeddings for question + each choice
        inputs = [f"{question} [SEP] {choice}" for choice in choices]
        # embeddings = self.model.encode(inputs, normalize_embeddings=True)

        return {
            "inputs": inputs,  # Shape: (num_choices,)
            "answer_idx": answer_idx,  # Correct choice index,
            "category": item["question_id"].split("_")[0],  # Question category
        }


# Define Model Class
class MultipleChoiceQAModel(nn.Module):
    def __init__(
        self, embedding_dim=384, model_name="sentence-transformers/all-mpnet-base-v2"
    ):
        """
        Args:
            embedding_dim: Dimensionality of SentenceTransformer embeddings (default is 384 for many models).
        """
        super(MultipleChoiceQAModel, self).__init__()
        self.model = SentenceTransformer(model_name)
        # freeze the SentenceTransformer model
        for param in self.model.parameters():
            param.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, embeddings):
        """
        Args:
            embeddings: Tensor of shape (batch_size * num_choices, embedding_dim)

        Returns:
            scores: Tensor of shape (batch_size, num_choices)
        """
        embeddings = self.model.encode(embeddings)
        embeddings = torch.tensor(embeddings).to(self.model.device)
        scores = self.mlp(embeddings)  # Shape: (batch_size * num_choices, 1)
        return scores.view(-1, 4)  # Reshape to (batch_size, num_choices)


# Training Function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", total=len(dataloader)):
        inputs = batch["inputs"]  # Shape: (batch_size, num_choices)
        # flatten list of lists
        inputs = [item for sublist in inputs for item in sublist]
        answer_idx = batch["answer_idx"].to(device)

        optimizer.zero_grad()

        scores = model(inputs)

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

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            inputs = batch["inputs"]  # Shape: (batch_size * num_choices, embedding_dim)
            inputs = [item for sublist in inputs for item in sublist]
            answer_idx = batch["answer_idx"].to(device)

            scores = model(
                inputs
            )  # Flatten to (batch_size * num_choices, embedding_dim)

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
    for category in category_total:
        category_accuracies[category] = (
            category_correct[category] / category_total[category]
        )
        print(f"{category} Accuracy: {category_accuracies[category]:.4f}")

    return accuracy, category_accuracies


# Main Script
if __name__ == "__main__":
    # Paths and Parameters
    train_json = "/data/user_data/gdhanuka/STAR_dataset/STAR_train.json"
    val_json = "/data/user_data/gdhanuka/STAR_dataset/STAR_val.json"
    batch_size = 16
    num_epochs = 50
    learning_rate = 5e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load SentenceTransformer Model
    sentence_transformer_model_name = "sentence-transformers/all-mpnet-base-v2"  # Replace with your desired SentenceTransformer model name

    # Load Datasets and Dataloaders
    train_dataset = TextOnlyQADataset(json_file=train_json)
    # train_dataset = torch.utils.data.Subset(train_dataset, range(1000))  # For demonstration purposes
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TextOnlyQADataset(json_file=val_json)
    # val_dataset = torch.utils.data.Subset(train_dataset, range(100))  # For demonstration purposes
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize Model and Optimizer
    qa_model = MultipleChoiceQAModel(
        embedding_dim=768, model_name=sentence_transformer_model_name
    ).to(
        device
    )  # Adjust embedding_dim if using a different SentenceTransformer model.
    optimizer = torch.optim.Adagrad(
        qa_model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0

    print("Hyperparameters:")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Device: {device}")
    print()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train(
            model=qa_model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_accuracy, _ = evaluate(
            model=qa_model, dataloader=val_dataloader, device=device
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"New Best Validation Accuracy: {best_val_accuracy:.4f}")
