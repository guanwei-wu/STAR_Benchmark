import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adagrad
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import random
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")
import os
import pickle
from sentence_transformers import SentenceTransformer
import av
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from huggingface_hub import hf_hub_download

class VideoQADataset(Dataset):
    def __init__(self, json_file, video_dir="/data/user_data/gdhanuka/STAR_dataset/Charades_v1_480", num_frames=8):
        """
        Args:
            json_file (str): Path to the JSON file containing the dataset.
            video_features_dir (str): Directory containing precomputed CLIP features for videos.
            num_frames (int): Number of frames to sample from each video.
        """
        with open(json_file, "rb") as f:
            self.data = pickle.load(f)
        self.video_dir = video_dir
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    def read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def __getitem__(self, idx):
        item = self.data[idx]
        video_id = item["video_id"]
        start, end = item["start"], item["end"]
        question = item["question"]
        choices = [choice["choice"] for choice in item["choices"]]
        answer_idx = next(
            i
            for i, choice in enumerate(item["choices"])
            if choice["choice"] == item["answer"]
        )

        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        frame_rate = container.streams.video[0].average_rate
        # use the part of the video between start and end
        indices = np.linspace(start*frame_rate, end*frame_rate, self.num_frames).astype(int)
        video_frames = self.read_video_pyav(container, indices)

        all_text_inputs = []
        for choice in choices:
            all_text_inputs.append(f"{question} [SEP] {choice}")

        return {
            "video_frames": video_frames,  # Video features
            "question": question,
            "choices": choices,
            "answer_idx": answer_idx,
            "category": item["question_id"].split("_")[0],  # Question category
            "all_text_inputs": all_text_inputs,
        }

class VideoQAModel():
    def __init__(self):
        self.model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2").to("cuda")
        self.processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

    def generate(self, inputs, max_new_tokens=500):
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def video_qa(self, video_frames, question, choices, max_new_tokens=500):
        choice_with_idx = [f"\"{i+1}\": {choice}\n" for i, choice in enumerate(choices)]
        prompt = f"USER: <video>\n According to the video choose the correct answer, {question} \n {choice_with_idx} ASSISTANT: "
        inputs = self.processor(text=prompt, videos=video_frames, return_tensors="pt").to("cuda")
        return self.generate(inputs, max_new_tokens=max_new_tokens)[0]

if __name__ == "__main__":
    val_pkl = "/data/user_data/gdhanuka/STAR_dataset/STAR_val.pkl"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = VideoQADataset(val_pkl, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    video_qa_model = VideoQAModel()

    for batch in tqdm(dataloader):
        video_frames = batch["video_frames"][0].to(device)
        question = batch["question"][0]
        choices = batch["choices"]
        answer_idx = batch["answer_idx"][0]
        category = batch["category"][0]
        all_text_inputs = batch["all_text_inputs"][0]
        # print(video_frames.shape, question, choices, answer_idx, category)
        answer = video_qa_model.video_qa(video_frames, question, choices)
        answer = int((answer.split("ASSISTANT: ")[1]).split("\n")[0])
        print(f"Question: {question}, Choices: {choices}, Predicted Index: {answer}, True index: {answer_idx+1}, True Answer: {choices[answer_idx]}")
        category_total[category] += 1
        if answer == answer_idx+1:
            category_correct[category] += 1
        print("Category-wise accuracy:")
        for category in category_total:
            print(f"{category}: {category_correct[category] / category_total[category]}")
        print(f"Overall accuracy: {sum(category_correct.values()) / sum(category_total.values())}")
        


