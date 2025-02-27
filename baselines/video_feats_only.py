# use the clip features with rnn, transformer and then mlp to predict one of the four choices

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import random
from collections import defaultdict


# Define Dataset Class
class VideoOnlyDataset(Dataset):
    def __init__(
        self,
        json_file,
        video_features_dir,
        num_frames=32,
    ):
        """
        Args:
            json_file (str): Path to the JSON file containing the dataset.
            video_features_dir (str): Directory containing precomputed RGB features for videos.
        """
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.video_features_dir = video_features_dir
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_id = item["video_id"]
        answer_idx = next(
            i
            for i, choice in enumerate(item["choices"])
            if choice["choice"] == item["answer"]
        )
        question_category = item["question_id"].split("_")[0]

        # Load precomputed RGB features for the video
        video_feats_path = f"{self.video_features_dir}/{video_id}_clip.npy"
        video_feats = np.load(video_feats_path)  # Shape: (num_frames, feature_dim)
        # print(video_feats.shape)

        total_frames = video_feats.shape[0]

        if total_frames == 0:
            # return zeros
            sampled_feats = np.zeros((self.num_frames, 512))
        elif total_frames >= self.num_frames:
            sampled_indices = sorted(
                random.sample(range(total_frames), self.num_frames)
            )
            sampled_feats = video_feats[sampled_indices]
        else:
            # If fewer frames than `num_frames`, repeat frames
            sampled_feats = np.repeat(
                video_feats, self.num_frames // total_frames + 1, axis=0
            )
            sampled_feats = sampled_feats[: self.num_frames]

        assert sampled_feats.shape == (self.num_frames, 512)

        sampled_feats = torch.from_numpy(sampled_feats).float()
        return {
            "video_feats": sampled_feats,  # Video features
            "answer_idx": answer_idx,  # Correct answer index
            "category": question_category,  # Question category
        }


# def collate_fn(batch):
#     """
#     Custom collate function to pad video features to the same number of frames in a batch.
#     """
#     max_frames = max(sample["video_feats"].shape[0] for sample in batch)
#     feature_dim = batch[0]["video_feats"].shape[1]

#     padded_video_feats = []
#     answer_idxs = []
#     categories = []

#     for sample in batch:
#         num_frames = sample["video_feats"].shape[0]
#         padding = torch.zeros(max_frames - num_frames, feature_dim)  # Pad with zeros
#         padded_video_feats.append(torch.cat([sample["video_feats"], padding], dim=0))
#         answer_idxs.append(sample["answer_idx"])
#         categories.append(sample["category"])

#     return {
#         "video_feats": torch.stack(padded_video_feats),  # Shape: (batch_size, max_frames, feature_dim)
#         "answer_idx": torch.tensor(answer_idxs),         # Shape: (batch_size,)
#         "category": categories                           # List of question categories
#     }


# Define RNN-Based Model
class VideoOnlyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(VideoOnlyModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_frames, input_dim)

        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        _, (hidden, _) = self.rnn(x)  # Use the final hidden state
        logits = self.fc(hidden[-1])  # Pass through fully connected layer
        return logits


# Training Function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, total=len(dataloader), desc="Training"):
        video_feats = batch["video_feats"].to(
            device
        )  # Shape: (batch_size, num_frames, feature_dim)
        answer_idx = batch["answer_idx"].to(device)  # Shape: (batch_size,)

        optimizer.zero_grad()

        logits = model(video_feats)  # Shape: (batch_size, num_classes)
        loss = criterion(logits, answer_idx)  # Cross-entropy loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss


# Evaluation Function
# Evaluation Function with Category-Wise Accuracy
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
            video_feats = batch["video_feats"].to(
                device
            )  # Shape: (batch_size * num_choices, seq_len)
            answer_idx = batch["answer_idx"].to(device)  # Shape: (batch_size,)
            categories = batch["category"]  # List of question categories

            logits = model(video_feats)  # Shape: (batch_size, num_classes)
            predictions = torch.argmax(logits, dim=1)  # Predicted class

            correct += (predictions == answer_idx).sum().item()
            total += len(answer_idx)

            # Update category-wise counts
            for i in range(len(categories)):
                category_correct[categories[i]] += int(predictions[i] == answer_idx[i])
                category_total[categories[i]] += 1

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Compute category-wise accuracy
    category_accuracies = {}
    for category in category_total:
        category_accuracies[category] = (
            category_correct[category] / category_total[category]
        )

    print("Category-Wise Accuracy:")
    for category, acc in category_accuracies.items():
        print(f"  {category}: {acc:.4f}")

    return accuracy, category_accuracies


# Main Script
if __name__ == "__main__":
    # Paths and Parameters
    train_json = "/data/user_data/gdhanuka/STAR_dataset/STAR_train.json"  # Replace with actual path to your dataset JSON file
    val_json = "/data/user_data/gdhanuka/STAR_dataset/STAR_val.json"  # Replace with actual path to your dataset JSON file
    clip_features_path = (
        "/data/user_data/gdhanuka/STAR_dataset/charades_clip_features/clip/ViT-B_32/"
    )
    video_features_dir = clip_features_path  # Directory with precomputed RGB features
    batch_size = 64
    input_dim = 512  # Feature dimension of precomputed RGB features
    hidden_dim = 512  # Hidden dimension of LSTM
    num_classes = 4  # Number of answer choices
    learning_rate = 1e-3
    num_epochs = 10  # Number of training epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Dataset and Dataloader
    dataset_train = VideoOnlyDataset(
        json_file=train_json, video_features_dir=video_features_dir
    )
    dataset_val = VideoOnlyDataset(
        json_file=val_json, video_features_dir=video_features_dir
    )

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size)

    print("Train data size = ", len(dataset_train))
    print("Val data size = ", len(dataset_val))

    # Initialize Model and Move to Device
    model = VideoOnlyModel(
        input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes
    ).to(device)

    # Define Optimizer and Loss Function
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train(
            model=model,
            dataloader=dataloader_train,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_accuracy = evaluate(model=model, dataloader=dataloader_val, device=device)
