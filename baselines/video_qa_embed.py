import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
import random
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
import os
import pickle
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Define Dataset Class
class VideoQADataset(Dataset):
    def __init__(self, json_file, video_features_dir, num_frames=16, preload=False):
        """
        Args:
            json_file (str): Path to the JSON file containing the dataset.
            video_features_dir (str): Directory containing precomputed CLIP features for videos.
            num_frames (int): Number of frames to sample from each video.
        """
        with open(json_file, "rb") as f:
            self.data = pickle.load(f)
        self.video_features_dir = video_features_dir
        self.num_frames = num_frames
        self.preload = preload

        if preload:
            self.video_features = {}
            for item in tqdm(self.data, desc="Preloading Video Features"):
                video_id = item["video_id"]
                video_feats_path = f"{self.video_features_dir}/{video_id}_clip.npy"
                self.video_features[video_id] = np.load(video_feats_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_id = item["video_id"]
        question = item["question"]
        choices = [choice["choice"] for choice in item["choices"]]
        answer_idx = next(
            i
            for i, choice in enumerate(item["choices"])
            if choice["choice"] == item["answer"]
        )

        if self.preload:
            video_feats = self.video_features[video_id]
        else:
            video_feats_path = f"{self.video_features_dir}/{video_id}_clip.npy"
            video_feats = np.load(
                video_feats_path
            )  # Shape: (num_frames_in_video, feature_dim)

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

        all_text_inputs = []
        for choice in choices:
            all_text_inputs.append(f"{question} [SEP] {choice}")

        return {
            "video_feats": sampled_feats,  # Video features
            "question": question,
            "choices": choices,
            "answer_idx": answer_idx,
            "category": item["question_id"].split("_")[0],  # Question category
            "all_text_inputs": all_text_inputs,
        }

def collate_fn(batch):
    """
    Custom collate function to handle batching of (question + choice) pairs.
    """
    video_feats = torch.stack([item["video_feats"] for item in batch])  # Shape: (batch_size, num_frames, feature_dim)
    questions = [item["question"] for item in batch]
    all_choices = [item["choices"] for item in batch]  # List of lists
    answer_idxs = torch.tensor([item["answer_idx"] for item in batch], dtype=torch.long)
    categories = [item["category"] for item in batch]
    all_text_inputs = []
    for item in batch:
        all_text_inputs = all_text_inputs + item["all_text_inputs"] # shape: (batch_size * num_choices,)

    return {
        "video_feats": video_feats,
        "question": questions,
        "choices": all_choices,
        "all_text_inputs": all_text_inputs,
        "answer_idx": answer_idxs,
        "category": categories,
    }

# Define Model for MCQA
class VideoQAModel(nn.Module):
    def __init__(self, text_model_name, input_dim, hidden_dim, text_hidden_dim, hidden_dim1, hidden_dim2):
        super(VideoQAModel, self).__init__()

        # Text Encoder (e.g., BERT)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        # Reduce text embedding dimension if needed
        self.text_fc = nn.Linear(self.text_encoder.config.hidden_size, text_hidden_dim)

        # Video Encoder (RNN)
        self.video_rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Fusion Layer and Scoring Layer
        self.fc_fusion1 = nn.Linear(hidden_dim + text_hidden_dim, hidden_dim1)  # Score per choice
        self.fc_fusion2 = nn.Linear(hidden_dim1, 1)  # Score per choice
        nn.init.xavier_uniform_(self.fc_fusion1.weight)
        nn.init.xavier_uniform_(self.fc_fusion2.weight)

    def forward(self, video_feats, text_input_ids, text_attention_mask):
        """
        Args:
            video_feats: Tensor of shape (batch_size * num_choices, num_frames, input_dim)
            text_input_ids: Tensor of shape (batch_size * num_choices, seq_len) [Tokenized question/choice]
            text_attention_mask: Tensor of shape (batch_size * num_choices, seq_len) [Attention mask for text]

        Returns:
            logits: Tensor of shape (batch_size, num_choices)
        """

        # Encode Text Features
        text_outputs = self.text_encoder(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        )

        text_embeds = self.text_fc(
            text_outputs.last_hidden_state[:, 0]
        )  # CLS token embedding

        # Encode Video Features
        _, (video_embeds, _) = self.video_rnn(video_feats)  # Use final hidden state
        video_embeds = video_embeds[-1]  # Shape: (batch_size * num_choices, hidden_dim)

        # check if repeated correctly
        assert video_embeds[0].equal(video_embeds[1])
        assert video_embeds[0].equal(video_embeds[2])
        assert video_embeds[0].equal(video_embeds[3])

        # Concatenate Text and Video Features
        combined_embeds = torch.cat(
            [video_embeds, text_embeds], dim=1
        )  # Shape: (batch_size * num_choices, hidden_dim + text_hidden_dim)

        # Scoring Layer (one score per choice)
        logits = self.fc_fusion2(self.fc_fusion1(combined_embeds)).squeeze(
            -1
        )  # Shape: (batch_size * num_choices,)

        return logits.view(-1, 4)  # Reshape to (batch_size, num_choices)


# Training Function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, total=len(dataloader), desc="Training"):
        # import ipdb; ipdb.set_trace()
        optimizer.zero_grad()

        # Video features
        video_feats = batch["video_feats"].to(device)  # Shape: (batch_size, num_frames, feature_dim)

        # Tokenize all (question + choice) pairs
        tokenized_inputs = tokenizer(
            batch["all_text_inputs"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        # Repeat video features for each choice
        batch_size = video_feats.size(0)
        num_choices = 4
        video_feats_repeated = video_feats.repeat_interleave(num_choices, dim=0)  # Shape: (batch_size * num_choices, num_frames, feature_dim)

        # Forward pass
        logits = model(
            video_feats=video_feats_repeated,
            text_input_ids=tokenized_inputs["input_ids"],
            text_attention_mask=tokenized_inputs["attention_mask"],
        )

        # Compute loss
        loss = criterion(logits, batch["answer_idx"].to(device))
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")



# Evaluation Function
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    with torch.no_grad():

        for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):

            video_feats = batch["video_feats"].to(device)
            answer_idx = batch["answer_idx"].to(device)
            categories = batch["category"]

            video_feats = batch["video_feats"].to(device)

            # Tokenize all (question + choice) pairs
            tokenized_inputs = tokenizer(
                batch["all_text_inputs"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            # Repeat video features for each choice
            batch_size = video_feats.size(0)
            num_choices = 4
            video_feats_repeated = video_feats.repeat_interleave(num_choices, dim=0)

            # Forward pass
            logits = model(
                video_feats=video_feats_repeated,
                text_input_ids=tokenized_inputs["input_ids"],
                text_attention_mask=tokenized_inputs["attention_mask"],
            )

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


if __name__ == "__main__":
    # Paths and Parameters
    torch.backends.cudnn.benchmark = True

    train_json = "/data/user_data/gdhanuka/STAR_dataset/STAR_train.json"  # Replace with actual path to your dataset JSON file
    val_json = "/data/user_data/gdhanuka/STAR_dataset/STAR_val.json"  # Replace with actual path to your dataset JSON file
    train_pkl = "/data/user_data/gdhanuka/STAR_dataset/STAR_train.pkl"
    val_pkl = "/data/user_data/gdhanuka/STAR_dataset/STAR_val.pkl"
    clip_features_path = (
        "/data/user_data/gdhanuka/STAR_dataset/charades_clip_features/clip/ViT-B_32/"
    )

    batch_size = 32
    input_dim = 512  # Feature dimension of precomputed CLIP features
    hidden_dim = 512  # Hidden dimension of LSTM
    text_hidden_dim = 512  # Reduced text embedding dimension
    num_sampled_frames = 64  # Number of frames to sample per video
    learning_rate = 1e-3
    num_epochs = 50  # Number of training epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Tokenizer for Text Encoder (e.g., BERT)
    text_model_name = "deepset/tinyroberta-squad2"  # You can replace this with any Hugging Face model
    tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)

    # Load Dataset and Dataloader
    dataset_train = VideoQADataset(
        json_file=train_pkl,
        video_features_dir=clip_features_path,
        num_frames=num_sampled_frames,
        preload=True
    )

    # dataset_train = torch.utils.data.Subset(dataset_train, range(2))

    dataset_val = VideoQADataset(
        json_file=val_pkl,
        video_features_dir=clip_features_path,
        num_frames=num_sampled_frames,
        preload=True
    )

    # dataset_val = torch.utils.data.Subset(dataset_val, range(100))
    # dataset_val = dataset_train

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Adjust based on your system (e.g., 4 workers for 1 GPU)
        pin_memory=True,
        collate_fn=collate_fn,
        # drop_last=True
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        num_workers=4,  # Adjust based on your system (e.g., 4 workers for 1 GPU)
        pin_memory=True,
        collate_fn=collate_fn,
        # drop_last=True
    )

    print("Train data size = ", len(dataset_train))
    print("Val data size = ", len(dataset_val))

    # Initialize Model and Move to Device
    model = VideoQAModel(
        text_model_name=text_model_name,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        text_hidden_dim=text_hidden_dim,
        hidden_dim1=256,
        hidden_dim2=32
    ).to(device)

    # Define Optimizer and Loss Function
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train(
            model=model,
            dataloader=dataloader_train,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        evaluate(model=model, dataloader=dataloader_val, device=device)
