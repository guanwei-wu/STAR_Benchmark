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
from time import perf_counter

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
            self.data = pickle.load(f)
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
        end = perf_counter()
        return {
            "video_frames": video_frames,  # Video features
            "question": question,
            "choices": choices,
            "answer_idx": answer_idx,
            "category": item["question_id"].split("_")[0],  # Question category
            "all_text_inputs": all_text_inputs,
            "data_proc_time": end-start,
            "question_id": question_id,
            "frame_ids": frame_idx
        }


class VideoQAModel:
    def __init__(self, load_in_bits=16):

        device = 'cuda'
        compute_dtype = 'fp16'
        double_quant = True
        quant_type = 'nf4'

        compute_dtype = (torch.float16 if compute_dtype == 'fp16' else (torch.bfloat16 if compute_dtype == 'bf16' else torch.float32))
        
        bnb_model_from_pretrained_args = {}
        if load_in_bits in [4, 8]:
            from transformers import BitsAndBytesConfig
            bnb_model_from_pretrained_args.update(dict(
                device_map={"": device},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=load_in_bits == 4,
                    load_in_8bit=load_in_bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=double_quant,
                    bnb_4bit_quant_type=quant_type # {'fp4', 'nf4'}
                )
            ))

        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            "LanguageBind/Video-LLaVA-7B-hf",
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2",
            **bnb_model_from_pretrained_args
        )
        self.processor = VideoLlavaProcessor.from_pretrained(
            "LanguageBind/Video-LLaVA-7B-hf"
        )

    def generate(self, inputs, max_new_tokens=20):  # Reduced from 500
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True, 
            output_scores=True,
        )
        print(outputs.sequences)
        decoded = self.processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        print(decoded)
        first_token_probs = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
        
        # Get token IDs for numbers 1-4
        token_ids = [self.processor.tokenizer.convert_tokens_to_ids(str(i)) for i in [1,2,3,4]]
        
        # Create probability dictionary for each sample in batch
        prob_list = []
        for batch_idx in range(first_token_probs.shape[0]):
            probs = [
                first_token_probs[batch_idx, token_ids[i]].item()
                for i in range(4)
            ]
            prob_list.append(probs)
        
        logits_list = []
        for batch_idx in range(outputs.scores[0].shape[0]):
            logits = [
                outputs.scores[0][batch_idx, token_ids[i]].item()
                for i in range(4)
            ]
            logits_list.append(logits)

        return decoded, prob_list, logits_list, outputs.sequences

    def video_qa_single(self, video_frames, question, choices, max_new_tokens=500):
        choice_with_idx = [f'"{i+1}": {choice}\n' for i, choice in enumerate(choices)]
        prompt = f"USER: <video>\n {question} \n {choice_with_idx} Answer with the option's index from the given choices directly. \n ASSISTANT: "
        inputs = self.processor(
            text=prompt, videos=video_frames, return_tensors="pt", max_length=4096
        ).to("cuda")
        decoded, probs, logits, _ = self.generate(inputs, max_new_tokens=max_new_tokens)
        return decoded[0], probs[0], prompt, logits[0]

    def video_qa_batch(self, video_frames_list, questions_list, choices_batch, max_new_tokens=20):
        # Create a list of prompts for each question and its choices
        
        print(choices_batch)
        
        prompts = []
        for question, choices in zip(questions_list, choices_batch):
            choice_with_idx = [f'"{i+1}": {choice}\n' for i, choice in enumerate(choices)]
            prompt = f"USER: <video>\n {question} \n {choice_with_idx} Answer with the option's index from the given choices directly. \n ASSISTANT: "
            prompts.append(prompt)
        
        self.processor.tokenizer.padding_side = "right"
        
        # Process inputs with padding
        inputs = self.processor(
            text=prompts,
            videos=video_frames_list,
            return_tensors="pt",
            padding=True,
            max_length=4096,
            truncation=True
        ).to("cuda")
        
        print('pixel_values_videos', inputs['pixel_values_videos'].shape, inputs['pixel_values_videos'])
        print('input_ids', inputs['input_ids'])
        print('attention_mask', inputs['attention_mask'])
        
        # Generate outputs
        decoded, probs, logits, _ = self.generate(inputs, max_new_tokens=max_new_tokens)
        
        # Return batch results
        return decoded, probs, prompts, logits


def list_collate_fn(batch):
    """Collate function that maintains all items as lists"""
    collated = defaultdict(list)
    for item in batch:
        for k, v in item.items():
            collated[k].append(v)
    return dict(collated)


if __name__ == "__main__":
    # Settings and paths
    val_pkl = "data/STAR_val.pkl"
    video_dir = "data/Charades_v1_480"
    results_file = "analysis/video_llava_results2.jsonl"
    final_accuracy_file = "analysis/video_llava_final_accuracy2.txt"
    
    # Set batch size to 1 for non-batch mode or higher for batch processing
    batch_size = 4  # Change to 1 for non-batch processing
    # use_batch_inference = batch_size > 1
    
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set up dataset and dataloader
    dataset = VideoQADataset(
        val_pkl, 
        video_dir=video_dir, 
        sampling_fps=4, 
        num_frames=8, 
        use_fps=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=list_collate_fn,
    )
    
    # Initialize counters
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    
    # Initialize model
    video_qa_model = VideoQAModel(load_in_bits=4)
    
    # Open results file in append mode
    with open(results_file, "a") as results_f:
        with tqdm(dataloader, desc="Evaluating", dynamic_ncols=True) as pbar:
            for batch in pbar:
                batch_size = len(batch["question"])
                start_time = perf_counter()
                
                # Get data
                data_time = batch["data_proc_time"]
                video_frames = batch["video_frames"]
                questions = batch["question"]
                choices_batch = batch["choices"]
                answer_idxs = batch["answer_idx"]
                categories = batch["category"]
                all_text_inputs = batch["all_text_inputs"]
                question_ids = batch["question_id"]
                frame_ids = batch["frame_ids"]
                
                # Handle batch vs single inference
                # if use_batch_inference:
                # Batch processing
                decoded_answers, probs_batch, prompts, logits_batch = video_qa_model.video_qa_batch(
                    video_frames, questions, choices_batch
                )
                
                print(decoded_answers)
                
                # Process each result in the batch
                for i in range(batch_size):
                    answer_raw = decoded_answers[i]
                    
                    # Extract the answer number using regex
                    try:
                        answer_match = re.search(r"(\d+)", answer_raw.split("ASSISTANT")[-1])
                        if answer_match:
                            answer = int(answer_match.group(1))
                        else:
                            # Fallback if regex fails
                            answer = 1  # Default answer if extraction fails
                            print(f"Warning: Could not extract answer number from: {answer_raw}")
                    except Exception as e:
                        print(f"Error extracting answer: {e}")
                        answer = 1
                        
                    # Process individual result
                    end_time = perf_counter()
                    
                    # Create JSON record for this sample
                    json_record = {
                        "question_id": question_ids[i],
                        "question": questions[i],
                        "choices": choices_batch[i],
                        "pred_ans_idx": answer - 1,
                        "true_index": answer_idxs[i],
                        "category": categories[i],
                        "raw_response": answer_raw,
                        "prompt": prompts[i],
                        "frame_ids": frame_ids[i].numpy().tolist(),
                        "inference_time": (end_time - start_time) / batch_size,  # Average time per example
                        "data_time": data_time[i],
                        "confidence": probs_batch[i],
                        "logits": logits_batch[i]
                    }
                    
                    # Write to results file
                    results_f.write(json.dumps(json_record) + "\n")
                    results_f.flush()
                    
                    # Update accuracy metrics
                    category_total[categories[i]] += 1
                    if answer == answer_idxs[i] + 1:
                        category_correct[categories[i]] += 1
                # else:
                #     # Single example processing (original approach)
                #     for i in range(batch_size):
                #         item_start_time = perf_counter()
                        
                #         answer_raw, probs, prompt, logits = video_qa_model.video_qa_single(
                #             video_frames[i], questions[i], choices_batch[i]
                #         )
                        
                #         # Extract the answer number
                #         try:
                #             answer = int(re.search(r"\d+", answer_raw.split("ASSISTANT")[-1]).group())
                #         except:
                #             answer = 1  # Default if extraction fails
                            
                #         item_end_time = perf_counter()
                        
                #         # Create JSON record
                #         json_record = {
                #             "question_id": question_ids[i],
                #             "question": questions[i],
                #             "choices": choices_batch[i],
                #             "pred_ans_idx": answer - 1,
                #             "true_index": answer_idxs[i].item(),
                #             "category": categories[i],
                #             "raw_response": answer_raw,
                #             "prompt": prompt,
                #             "frame_ids": frame_ids[i].numpy().tolist(),
                #             "inference_time": (item_end_time - item_start_time),
                #             "data_time": data_time[i],
                #             "confidence": probs,
                #             "logits": logits
                #         }
                        
                #         # Write to results file
                        results_f.write(json.dumps(json_record) + "\n")
                        results_f.flush()
                        
                        # Update accuracy metrics
                        category_total[categories[i]] += 1
                        if answer == answer_idxs[i] + 1:
                            category_correct[categories[i]] += 1
                
                # Compute overall accuracy for the progress bar
                overall_acc = sum(category_correct.values()) / max(sum(category_total.values()), 1)
                
                # Update tqdm bar with accuracy info
                accuracy_info = {cat: f"{category_correct[cat] / max(category_total[cat], 1):.3f}" for cat in category_total}
                accuracy_info["Overall"] = f"{overall_acc:.3f}"
                pbar.set_postfix(accuracy_info)
                
    # Save final category-wise accuracy to a text file
    with open(final_accuracy_file, "w") as acc_f:
        acc_f.write("Final Category-Wise Accuracy:\n")
        for category in category_total:
            acc_f.write(f"{category}: {category_correct[category] / category_total[category]:.4f}\n")
        acc_f.write(f"\nOverall accuracy: {sum(category_correct.values()) / sum(category_total.values()):.4f}\n")

    print(f"Results saved to {results_file} and final accuracy saved to {final_accuracy_file}")