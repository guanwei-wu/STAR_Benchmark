import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adagrad
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
import random
from collections import defaultdict
import warnings
import argparse
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
from huggingface_hub import hf_hub_download

from video_qa_dataset import VideoQADataset
from qwen_vl_utils import process_vision_info

class QwenVideoQA_CoT:
    def __init__(self, num_frames):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        self.num_frames = num_frames

    def generate(self, inputs, max_new_tokens=500):
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            return_dict_in_generate=True, output_scores=True
        )
        sequence_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs.sequences)
        ]

        decoded = self.processor.batch_decode(
            sequence_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # Get the last non-EOS token probabilities for CoT
        last_non_eos_token_probs = torch.nn.functional.softmax(outputs.scores[-2], dim=-1)
        
        # Get token IDs for numbers 1-4
        token_ids = [self.processor.tokenizer.convert_tokens_to_ids(str(i)) for i in [1,2,3,4]]
        
        # Create probability dictionary for each sample in batch
        prob_list = []
        for batch_idx in range(last_non_eos_token_probs.shape[0]):
            probs = [
                last_non_eos_token_probs[batch_idx, token_ids[i]].item()
                for i in range(4)
            ]
            prob_list.append(probs)
        
        logits_list = []
        for batch_idx in range(outputs.scores[-2].shape[0]):
            logits = [
                outputs.scores[-2][batch_idx, token_ids[i]].item()
                for i in range(4)
            ]
            logits_list.append(logits)

        return decoded, prob_list, logits_list

    def video_qa_cot(self, video_path, question, choices, start, end, max_new_tokens=500):
        choice_with_idx = ''.join([f"{i+1}. {choice[0]}" + chr(10) for i, choice in enumerate(choices)])
        
        prompt = f"Question: {question}\nChoices:\n{choice_with_idx}\nPlease carefully watch the video and reason through the visual content to answer the question based on the given choices.\n\nFirst, explain your reasoning in a few sentences describing what you observed in the video and how it relates to the question. Finally, you must state your final answer using the format:\nAnswer: <index>"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "video_start": start,
                        "video_end": end,
                        "nframes": self.num_frames
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        decoded, probs, logits = self.generate(inputs, max_new_tokens=max_new_tokens)
        
        return decoded[0], probs[0], prompt, logits[0]


def extract_answer_index(answer_raw):
    # Try matching "Answer: <index>" or "Answer :<index>" (case-insensitive)
    match = re.search(r"(?i)answer\s*[:\-]?\s*(\d+)", answer_raw)
    if match:
        return int(match.group(1))

    # Try matching "is <index>"
    match = re.search(r"\bis\s+(\d+)", answer_raw)
    if match:
        return int(match.group(1))

    # Try matching the last number in the string
    match = re.search(r"(\d+)(?!.*\d)", answer_raw)
    if match:
        return int(match.group(1))

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen VL STAR Benchmark Inference with Chain-of-Thought")

    parser.add_argument("--val_pkl", type=str, default="/data/user_data/gdhanuka/STAR_dataset/STAR_val.json", help="Path to validation .pkl file")
    parser.add_argument("--video_dir", type=str, default="/data/user_data/gdhanuka/STAR_dataset/Charades_v1_480", help="Path to video directory")
    parser.add_argument("--results_file", type=str, default="/home/gdhanuka/STAR_Benchmark/analysis/qwen2vl_cot_results.jsonl", help="Path to write model predictions")
    parser.add_argument("--final_accuracy_file", type=str, default="/home/gdhanuka/STAR_Benchmark/analysis/qwen2vl_cot_final_accuracy.txt", help="Path to write final accuracy results")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of video frames to sample for inference")

    args = parser.parse_args()

    # Now use these variables below
    val_pkl = args.val_pkl
    video_dir = args.video_dir
    results_file = args.results_file
    final_accuracy_file = args.final_accuracy_file
    num_frames = args.num_frames

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = VideoQADataset(val_pkl, video_dir=video_dir, sampling_fps=4, num_frames=num_frames, use_fps=False)
    # batched inference not working!!
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,  # randomize to get a good sample for evaluation
        num_workers=4,
        pin_memory=True,
    )
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    video_qa_model = QwenVideoQA_CoT(num_frames=num_frames)

    # Open results file in append mode
    with open(results_file, "a") as results_f:
        with tqdm(dataloader, desc="Evaluating", dynamic_ncols=True) as pbar:
            for batch in pbar:
                start_time = perf_counter()
                data_time = batch["data_proc_time"][0]
                start = batch["start"][0].item()
                end = batch["end"][0].item()
                video_frames = batch["video_frames"][0].to(device)
                question = batch["question"][0]
                choices = batch["choices"]
                choices = [choice for choice in choices]  # keep as list of lists for compatibility
                answer_idx = batch["answer_idx"][0]
                category = batch["category"][0]
                all_text_inputs = batch["all_text_inputs"][0]
                question_id = batch["question_id"][0]
                frame_ids = batch["frame_ids"][0]
                video_path = batch["video_path"][0]

                # Get model prediction
                answer_raw, probs, prompt, logits = video_qa_model.video_qa_cot(video_path, question, choices, start, end)
                
                match = extract_answer_index(answer_raw)

                if match:
                    answer = match
                else:
                    print(f"Error parsing answer: {answer_raw}")
                    answer = 1  # default answer
                    
                end_time = perf_counter()

                # Save result to JSONL file
                json_record = {
                    "question_id": question_id,
                    "question": question,
                    "choices": [c[0] for c in choices],
                    "pred_ans_idx": answer-1,
                    "true_index": (answer_idx).item(),  # a tensor
                    "category": category,
                    "raw_response": answer_raw,
                    "prompt": prompt,
                    "frame_ids": frame_ids.numpy().tolist(),
                    "inference_time": (end_time - start_time),
                    "data_time": data_time.item(),
                    "confidence": probs,
                    "logits": logits
                }

                results_f.write(json.dumps(json_record) + "\n")  # Append each example as a new line
                results_f.flush()

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

"""

python baselines/qwen2vl_cot.py \
    --val_pkl "/data/user_data/jamesdin/STAR/data/STAR_val.json" \
    --video_dir "/data/user_data/jamesdin/STAR/data/Charades_v1_480" \
    --results_file "analysis/qwen2vl_cot_results.jsonl" \
    --final_accuracy_file "analysis/qwen2vl_cot_final_accuracy.txt" \
    --num_frames 8
"""