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
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import re
import bisect
import shutil
import json

# warnings.filterwarnings("ignore")
import os
import pickle
from sentence_transformers import SentenceTransformer
import av
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from huggingface_hub import hf_hub_download


class VideoQADataset(Dataset):
    def __init__(
        self,
        json_file,
        video_dir="/data/user_data/gdhanuka/STAR_dataset/Charades_v1_480",
        sampling_fps=8,
    ):
        """
        Args:
            json_file (str): Path to the JSON file containing the dataset.
            video_features_dir (str): Directory containing precomputed CLIP features for videos.
            num_frames (int): Number of frames to sample from each video.
        """
        with open(json_file, "rb") as f:
            self.data = pickle.load(f)
        self.video_dir = video_dir
        self.sampling_fps = sampling_fps

    def __len__(self):
        return len(self.data)

    def read_video_pyav(self, container, indices):
        """
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        """
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

    def read_video_pyav2(self, video_path, start, end, num_frames=8):
        """Reads a video for given start-end timestamps interval and uniformly samples 8 frames of it"""
        container = av.open(video_path)
        video = container.streams.get(0)[0]

        av_timestamps = [
            int(packet.pts * video.time_base)
            for packet in container.demux(video)
            if packet.pts is not None
        ]

        av_timestamps.sort()
        start_id = bisect.bisect_left(av_timestamps, start)
        end_id = bisect.bisect_left(av_timestamps, end)

        # in case it is a very short video, lets take a longer duration and sample
        if end_id - start_id < 10:
            end_id += 10
            start_id -= 10

        end_id = min(len(av_timestamps) - 1, end_id)
        start_id = max(1, start_id)

        # We sample 8 frames for tuning following the original paper
        # But we can increase the number of frames for longer videos and check out if it helps performance
        # Change the below "8" to any number of frames you want, and note that more frames -> more computational resources needed
        indices = np.linspace(start_id, end_id, num_frames).astype(int)

        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_id:
                break
            if i >= start_id and i in indices:
                frames.append(frame)
        assert (
            len(frames) == num_frames
        ), f"Got {len(frames)} frames but should be {num_frames}. Check the indices: {indices};, start_id: {start_id}, end_id: {end_id}. Len of video is {len(av_timestamps)} frames."
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
        # container = av.open(video_path)
        # total_frames = container.streams.video[0].frames
        # frame_rate = container.streams.video[0].average_rate
        # clip_duration = end - start
        # num_frames = int(self.sampling_fps * clip_duration)
        # # use the part of the video between start and end
        # indices = np.linspace(
        #     start * frame_rate, end * frame_rate, num_frames
        # ).astype(int)
        # video_frames = self.read_video_pyav(container, indices)
        # video_frames = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float()

        video_frames = self.read_video_pyav2(video_path, start, end, num_frames=16)
        video_frames = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float()

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


def collate_fn(batch):
    """Handles variable-sized video frames using smart padding"""
    # Separate video frames and metadata
    videos = [item["video_frames"] for item in batch]
    questions = [item["question"] for item in batch]
    choices = [item["choices"] for item in batch]
    answer_idxs = torch.stack([torch.tensor(item["answer_idx"]) for item in batch])

    # Pad videos to max dimensions in batch
    max_frames = max(vid.shape[0] for vid in videos)
    max_height = max(vid.shape[2] for vid in videos)
    max_width = max(vid.shape[3] for vid in videos)

    padded_videos = []
    for vid in videos:
        # Pad: (width_left, width_right, height_top, height_bottom, frames_front, frames_back)
        pad_width = max_width - vid.shape[3]
        pad_height = max_height - vid.shape[2]
        pad_frames = max_frames - vid.shape[0]

        padded = F.pad(vid, (0, pad_width, 0, pad_height, 0, 0, 0, pad_frames))
        padded_videos.append(padded)

    return {
        "video_frames": torch.stack(padded_videos),
        "question": questions,
        "choices": choices,
        "answer_idx": answer_idxs,
    }


class VideoQAModel:
    def __init__(self):
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            "LanguageBind/Video-LLaVA-7B-hf",
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).to("cuda")
        self.processor = VideoLlavaProcessor.from_pretrained(
            "LanguageBind/Video-LLaVA-7B-hf"
        )

    def generate(self, inputs, max_new_tokens=500):
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    def video_qa(self, video_frames, question, choices, max_new_tokens=500):
        choice_with_idx = [f'"{i+1}": {choice}\n' for i, choice in enumerate(choices)]
        prompt = f"USER: <video>\n According to the video choose the correct answer, {question} \n {choice_with_idx} ASSISTANT: "
        inputs = self.processor(
            text=prompt, videos=video_frames, return_tensors="pt"
        ).to("cuda")
        return self.generate(inputs, max_new_tokens=max_new_tokens)[0]

    def video_qa_batch(self, video_batch, questions, choices_batch):
        prompts = []
        for q, choices in zip(questions, choices_batch):
            opts = "\n".join([f"{i+1}: {c}" for i, c in enumerate(choices)])
            prompts.append(
                f"USER: <video>\n According to the video choose the correct answer, {questions} \n {opts} ASSISTANT: "
            )

        inputs = self.processor(
            text=prompts,
            videos=[v for v in video_batch],  # Process full batch
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to("cuda", torch.float16)

        outputs = self.model.generate(**inputs, max_new_tokens=20)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    # val_pkl = "/data/user_data/gdhanuka/STAR_dataset/STAR_val.pkl"
    val_pkl = "data/STAR_val.pkl"
    video_dir = "data/Charades_v1_480"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = VideoQADataset(val_pkl, sampling_fps=4, video_dir=video_dir)
    # batched inference not working!!
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # disable to easily reproduce
        num_workers=4,
        pin_memory=True,
        # collate_fn=collate_fn,
    )
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    video_qa_model = VideoQAModel()

    import json
    from tqdm import tqdm

    # File paths
    results_file = "analysis/video_llava_results.jsonl"
    final_accuracy_file = "analysis/video_llava_final_accuracy.txt"

    # Open results file in append mode
    with open(results_file, "a") as results_f:
        with tqdm(dataloader, desc="Evaluating", dynamic_ncols=True) as pbar:
            for batch in pbar:
                video_frames = batch["video_frames"][0].to(device)
                question = batch["question"][0]
                choices = batch["choices"]
                answer_idx = batch["answer_idx"][0]
                category = batch["category"][0]
                all_text_inputs = batch["all_text_inputs"][0]

                # Get model prediction
                answer = video_qa_model.video_qa(video_frames, question, choices)
                answer = int(re.search(r"\d+", answer.split("ASSISTANT")[-1]).group())

                # Save result to JSONL file
                json_record = {
                    "question": question,
                    "choices": choices,
                    "predicted_index": answer,
                    "true_index": (answer_idx + 1).item(),  # a tensor
                    "category": category
                }
                results_f.write(json.dumps(json_record) + "\n")  # Append each example as a new line

                # Update accuracy counters
                category_total[category] += 1
                if answer == answer_idx + 1:
                    category_correct[category] += 1

                # Compute overall accuracy
                overall_acc = sum(category_correct.values()) / sum(category_total.values())

                # Update tqdm bar with accuracy
                accuracy_info = {cat: f"{category_correct[cat] / category_total[cat]:.3f}" for cat in category_total}
                accuracy_info["Overall"] = f"{overall_acc:.3f}"
                pbar.set_postfix(accuracy_info)

    # Save final category-wise accuracy to a text file
    with open(final_accuracy_file, "w") as acc_f:
        acc_f.write("Final Category-Wise Accuracy:\n")
        for category in category_total:
            acc_f.write(f"{category}: {category_correct[category] / category_total[category]:.4f}\n")
        acc_f.write(f"\nOverall accuracy: {sum(category_correct.values()) / sum(category_total.values()):.4f}\n")

    print(f"Results saved to {results_file} and final accuracy saved to {final_accuracy_file}")


    # for batch in tqdm(dataloader):
    #     video_batch = batch["video_frames"].to("cuda", non_blocking=True)
    #     print(video_batch.shape)
    #     answers = video_qa_model.video_qa_batch(
    #         video_batch,
    #         batch["question"],
    #         batch["choices"]
    #     )

    #     # Batch evaluation
    #     for ans, true_idx, choices, cat in zip(answers,
    #                                          batch["answer_idx"],
    #                                          batch["choices"],
    #                                          batch["category"]):
    #         pred_idx = int(re.search(r'\d+', ans.split("ASSISTANT")[-1]).group())  # Add your parsing logic
    #         print(f"Predicted Index: {pred_idx}, True index: {true_idx+1}, True Answer: {choices[true_idx]}")
    #         category_total[cat] += 1
    #         if pred_idx == true_idx+1:
    #             category_correct[cat] += 1

    #     print("Category-wise accuracy:")
    #     for category in category_total:
    #         print(f"{category}: {category_correct[category] / category_total[category]}")
    #     print(f"Overall accuracy: {sum(category_correct.values()) / sum(category_total.values())}")
