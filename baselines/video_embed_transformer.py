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
from sentence_transformers import SentenceTransformer

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
        question_id = item["question_id"]
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
            sampled_indices = []
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
            sampled_indices = list(range(total_frames))

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
            "question_id": question_id,
            "frame_ids": sampled_indices
        }


def collate_fn(batch):
    """
    Custom collate function to handle batching of (question + choice) pairs.
    """
    video_feats = torch.stack(
        [item["video_feats"] for item in batch]
    )  # Shape: (batch_size, num_frames, feature_dim)
    questions = [item["question"] for item in batch]
    all_choices = [item["choices"] for item in batch]  # List of lists
    answer_idxs = torch.tensor([item["answer_idx"] for item in batch], dtype=torch.long)
    categories = [item["category"] for item in batch]
    all_text_inputs = []
    for item in batch:
        all_text_inputs = (
            all_text_inputs + item["all_text_inputs"]
        )  # shape: (batch_size * num_choices,)

    return {
        "video_feats": video_feats,
        "question": questions,
        "choices": all_choices,
        "all_text_inputs": all_text_inputs,
        "answer_idx": answer_idxs,
        "category": categories,
        "question_id": [item["question_id"] for item in batch],
        "frame_ids": [item["frame_ids"] for item in batch],
    }


class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        """
        Cross-Attention module for combining text and video features.
        Args:
            embed_dim (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate for attention weights.
        """
        super(CrossAttentionModule, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Forward pass for cross-attention.
        Args:
            query (Tensor): Query tensor (e.g., text embeddings) of shape (seq_len_q, batch_size, embed_dim).
            key (Tensor): Key tensor (e.g., video embeddings) of shape (seq_len_kv, batch_size, embed_dim).
            value (Tensor): Value tensor (e.g., video embeddings) of shape (seq_len_kv, batch_size, embed_dim).
        Returns:
            Tensor: Output after cross-attention and feed-forward layers.
        """
        # Cross-Attention
        attn_output, _ = self.multihead_attn(query, key, value)
        query = query + self.dropout(attn_output)  # Residual connection
        query = self.layer_norm(query)

        # Feed-Forward Network
        ffn_output = self.ffn(query)
        output = query + self.dropout(ffn_output)  # Residual connection
        output = self.layer_norm(output)

        return output


class VideoQAModelWithCrossAttention(nn.Module):
    def __init__(
        self,
        text_model_name,
        video_embed_dim=512,
        text_embed_dim=768,
        hidden_dim=512,
        num_heads=8,
    ):
        """
        Video Question Answering Model with Cross-Attention.
        Args:
            text_model_name (str): Name of the pre-trained SentenceTransformer model for text encoding.
            video_embed_dim (int): Dimension of video embeddings.
            text_embed_dim (int): Dimension of text embeddings from SentenceTransformer.
            hidden_dim (int): Hidden dimension for cross-attention and feed-forward layers.
            num_heads (int): Number of attention heads in cross-attention.
        """
        super(VideoQAModelWithCrossAttention, self).__init__()

        # Text Encoder
        self.text_encoder = SentenceTransformer(text_model_name)
        # freeze the text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Reduce dimensions if needed
        self.text_fc = nn.Linear(text_embed_dim, hidden_dim)

        # Video Encoder (LSTM)
        self.video_rnn = nn.LSTM(video_embed_dim, hidden_dim, batch_first=True)

        # Cross-Attention Module
        self.cross_attention = CrossAttentionModule(hidden_dim, num_heads=num_heads)

        # Scoring Layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, video_feats, text_inputs):
        """
        Forward pass for the model.

        Args:
            video_feats: Tensor of shape (batch_size * num_choices, num_frames, video_embed_dim).
            text_inputs: List of strings containing question-choice pairs.

        Returns:
            logits: Tensor of shape (batch_size, num_choices).
        """

        # Encode Text Features
        text_embeds = torch.tensor(
            self.text_encoder.encode(text_inputs), dtype=torch.float32
        ).to(video_feats.device)

        # Reduce Text Embedding Dimensions
        text_embeds = self.text_fc(
            text_embeds
        )  # Shape: (batch_size * num_choices, hidden_dim)

        # Reshape Text Embeddings for Attention
        text_embeds = text_embeds.unsqueeze(
            0
        )  # Shape: (1, batch_size * num_choices, hidden_dim)

        # Encode Video Features
        _, (video_embeds, _) = self.video_rnn(
            video_feats
        )  # Use final hidden state from LSTM
        video_embeds = video_embeds[-1]  # Shape: (batch_size * num_choices, hidden_dim)

        # Reshape Video Embeddings for Attention
        video_embeds = video_embeds.unsqueeze(
            0
        )  # Shape: (1, batch_size * num_choices, hidden_dim)

        # Cross-Attention between Text and Video Features
        attended_features = self.cross_attention(
            text_embeds, video_embeds, video_embeds
        )

        # Remove sequence dimension after attention
        attended_features = attended_features.squeeze(
            0
        )  # Shape: (batch_size * num_choices, hidden_dim)

        # Scoring Layer
        logits = self.mlp(attended_features).squeeze(
            -1
        )  # Shape: (batch_size * num_choices)

        return logits.view(-1, 4)  # Reshape to (batch_size, num_choices)


# Training Function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        # Video features
        video_feats = batch["video_feats"].to(device)

        # # Tokenize all (question + choice) pairs
        # tokenized_inputs = tokenizer(
        #     batch["all_text_inputs"], padding=True, truncation=True, return_tensors="pt"
        # ).to(device)

        # Repeat video features for each choice
        batch_size = video_feats.size(0)
        num_choices = 4
        video_feats_repeated = video_feats.repeat_interleave(num_choices, dim=0)

        # Forward pass
        logits = model(
            video_feats=video_feats_repeated,
            text_inputs=batch["all_text_inputs"],
        )

        # Compute loss
        loss = criterion(logits, batch["answer_idx"].to(device))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")


# Evaluation Function
def evaluate(model, dataloader, device, results_file):
    model.eval()
    correct = 0
    total = 0

    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    with open(results_file, "w") as results_f:
        with torch.no_grad():
            with tqdm(dataloader, total=len(dataloader), desc="Evaluating", dynamic_ncols=True) as pbar:
                for batch in pbar:

                    video_feats = batch["video_feats"].to(device)
                    answer_idx = batch["answer_idx"].to(device)
                    categories = batch["category"]

                    video_feats = batch["video_feats"].to(device)

                    # Tokenize all (question + choice) pairs
                    # tokenized_inputs = tokenizer(
                    #     batch["all_text_inputs"],
                    #     padding=True,
                    #     truncation=True,
                    #     return_tensors="pt",
                    # ).to(device)

                    # Repeat video features for each choice
                    batch_size = video_feats.size(0)
                    num_choices = 4
                    video_feats_repeated = video_feats.repeat_interleave(num_choices, dim=0)

                    # Forward pass
                    logits = model(
                        video_feats=video_feats_repeated,
                        text_inputs=batch["all_text_inputs"],
                    )

                    predictions = torch.argmax(logits, dim=1)  # Predicted class
                    probabilities = nn.functional.softmax(logits, dim=1)

                    correct += (predictions == answer_idx).sum().item()
                    total += len(answer_idx)

                    # Update category-wise counts
                    for i in range(len(categories)):
                        category_correct[categories[i]] += int(predictions[i] == answer_idx[i])
                        category_total[categories[i]] += 1

                    for i in range(len(predictions)):
                        # # for each item in batch, print the question_id, the pred ans, the true ans, probabilities and logits
                        # print(
                        #     f"Question ID: {batch['question_id'][i]}, Pred Answer: {predictions[i]}, True Answer: {answer_idx[i]}, Probabilities: {probabilities[i].cpu().numpy().tolist()}, Logits: {logits[i].cpu().numpy().tolist()}"
                        # )
                        json_record = {
                            "question_id": batch["question_id"][i],
                            "pred_ans_idx": predictions[i].item(),
                            "true_ans_idx": answer_idx[i].item(),
                            "confidence": probabilities[i].cpu().numpy().tolist(),
                            "logits": logits[i].cpu().numpy().tolist(),
                            "frame_ids": batch["frame_ids"][i],
                        }

                        results_f.write(json.dumps(json_record) + "\n")  # Append each example as a new line
                        results_f.flush()

                    # Compute overall accuracy
                    overall_acc = sum(category_correct.values()) / sum(category_total.values())

                    # Update tqdm bar with accuracy
                    accuracy_info = {cat: f"{category_correct[cat] / category_total[cat]:.3f}" for cat in category_total}
                    accuracy_info["Overall"] = f"{overall_acc:.3f}"
                    pbar.set_postfix(accuracy_info)


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
    input_dim = 512  # CLIP feature dimension
    hidden_dim = 512  # Hidden dimension for transformer
    num_heads = 4  # Number of attention heads in transformer layers
    num_layers = 2  # Number of transformer layers
    learning_rate = 1e-3
    num_epochs = 10
    num_sampled_frames = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Tokenizer for Text Encoder (e.g., BERT)
    text_model_name = "sentence-transformers/all-mpnet-base-v2"
    # tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)

    # Load Dataset and Dataloader
    dataset_train = VideoQADataset(
        json_file=train_pkl,
        video_features_dir=clip_features_path,
        num_frames=num_sampled_frames,
        preload=True,
    )
    # dataset_train = torch.utils.data.Subset(dataset_train, range(1000))  # For demonstration purposes

    dataset_val = VideoQADataset(
        json_file=val_pkl,
        video_features_dir=clip_features_path,
        num_frames=num_sampled_frames,
        preload=True,
    )
    # dataset_val = torch.utils.data.Subset(dataset_train, range(100))  # For demonstration purposes

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
    model = VideoQAModelWithCrossAttention(
        text_model_name=text_model_name,
        video_embed_dim=512,
        text_embed_dim=768,
        hidden_dim=hidden_dim,
        num_heads=8,
    ).to("cuda")

    # Define Optimizer and Loss Function
    optimizer = torch.optim.Adagrad(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0

    results_file = "videotf_64f2.jsonl"

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

        val_acc, _ = evaluate(model=model, dataloader=dataloader_val, device=device, results_file=results_file)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}")
