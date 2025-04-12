import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adagrad
from tqdm import tqdm
import torch
import numpy as np
import bisect
import shutil
import json
from time import perf_counter

# warnings.filterwarnings("ignore")
import os
import pickle
import av


class VideoQADataset(Dataset):
    def __init__(
        self,
        json_file,
        video_dir="/data/user_data/gdhanuka/STAR_dataset/Charades_v1_480",
        sampling_fps=4,
        num_frames=8,
        use_fps=True
    ):
        """
        Args:
            json_file (str): Path to the JSON file containing the dataset.
            video_features_dir (str): Directory containing precomputed CLIP features for videos.
            num_frames (int): Number of frames to sample from each video.
        """
        with open(json_file, "rb") as f:
            self.data = json.load(f)
        self.video_dir = video_dir
        self.sampling_fps = sampling_fps
        self.num_frames = num_frames
        self.use_fps = use_fps

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
        return np.stack([x.to_ndarray(format="rgb24") for x in frames]), indices

    def read_video_pyav3(self, video_path, start, end, sampling_fps=4):
        """Reads a video clip from start-end timestamps and samples frames at specified FPS"""
        container = av.open(video_path)
        video = container.streams.video[0]
        
        # Calculate number of frames needed based on duration and sampling FPS
        duration = end - start
        num_frames = int(round(sampling_fps * duration))
        num_frames = max(1, num_frames)  # Ensure at least 1 frame

        # Get sorted presentation timestamps
        av_timestamps = [
            int(packet.pts * video.time_base)
            for packet in container.demux(video)
            if packet.pts is not None
        ]
        av_timestamps.sort()

        # Find frame indices bounding our clip
        start_id = bisect.bisect_left(av_timestamps, start)
        end_id = bisect.bisect_left(av_timestamps, end)

        # Expand window for short clips
        if end_id - start_id < 10:
            end_id += 10
            start_id -= 10

        # Clamp to valid range
        end_id = min(len(av_timestamps) - 1, end_id)
        start_id = max(0, start_id)

        # Generate sampling indices
        indices = np.linspace(start_id, end_id, num_frames, dtype=int)

        # Extract frames
        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_id:
                break
            if i >= start_id and i in indices:
                frames.append(frame)
        
        assert len(frames) == num_frames, (
            f"Frame sampling failed: Expected {num_frames}, got {len(frames)}. "
            f"Time range: {start}-{end}s ({duration}s), Sampling FPS: {sampling_fps}."
        )
        
        return np.stack([x.to_ndarray(format="rgb24") for x in frames]), indices


    def __getitem__(self, idx):
        
        item = self.data[idx]
        video_id = item["video_id"]
        start, end = item["start"], item["end"]
        question = item["question"]
        question_id = item["question_id"]
        choices = [choice["choice"] for choice in item["choices"]]
        answer_idx = next(
            i
            for i, choice in enumerate(item["choices"])
            if choice["choice"] == item["answer"]
        )

        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        start_time = perf_counter()
        if self.use_fps:
            video_frames, frame_idx = self.read_video_pyav3(video_path, start, end, sampling_fps=self.sampling_fps)
        else:
            video_frames, frame_idx = self.read_video_pyav2(video_path, start, end, num_frames=self.num_frames)
        video_frames = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float() # (#frames, channel, h, w)

        all_text_inputs = []
        for choice in choices:
            all_text_inputs.append(f"{question} [SEP] {choice}")
        end_time = perf_counter()
        return {
            "video_frames": video_frames,  # Video features
            "question": question,
            "choices": choices,
            "answer_idx": answer_idx,
            "category": item["question_id"].split("_")[0],  # Question category
            "all_text_inputs": all_text_inputs,
            "data_proc_time": end_time-start_time,
            "start": start,
            "end": end,
            "question_id": question_id,
            "frame_ids": frame_idx,
            "video_path": video_path,
        }


# def collate_fn(batch):
#     """Handles variable-sized video frames using smart padding"""
#     # Separate video frames and metadata
#     videos = [item["video_frames"] for item in batch]
#     questions = [item["question"] for item in batch]
#     choices = [item["choices"] for item in batch]
#     answer_idxs = torch.stack([torch.tensor(item["answer_idx"]) for item in batch])

#     # Pad videos to max dimensions in batch
#     max_frames = max(vid.shape[0] for vid in videos)
#     max_height = max(vid.shape[2] for vid in videos)
#     max_width = max(vid.shape[3] for vid in videos)

#     padded_videos = []
#     for vid in videos:
#         # Pad: (width_left, width_right, height_top, height_bottom, frames_front, frames_back)
#         pad_width = max_width - vid.shape[3]
#         pad_height = max_height - vid.shape[2]
#         pad_frames = max_frames - vid.shape[0]

#         padded = F.pad(vid, (0, pad_width, 0, pad_height, 0, 0, 0, pad_frames))
#         padded_videos.append(padded)

#     return {
#         "video_frames": torch.stack(padded_videos),
#         "question": questions,
#         "choices": choices,
#         "answer_idx": answer_idxs,
#     }
