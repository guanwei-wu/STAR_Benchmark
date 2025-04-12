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

class VideoOfThoughtPredictor:
    def __init__(self, num_frames):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        self.num_frames = num_frames
        
        self.show_intermediate_steps = False
        
    def _generate_response(self, video_path, prompt, start, end, max_new_tokens=100):
        """
        Generate a response from the Qwen-VL model.
        
        Args:
            video_path: Path to the video file
            prompt: Text prompt
            start: Start time for video segment
            end: End time for video segment
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
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
        
        # Generate the response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

        sequence_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        
        decoded = self.processor.batch_decode(
            sequence_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        response = decoded[0]
        
        # Extract just the assistant's response, removing the prompt
        if self.show_intermediate_steps:
            print('#' * 50)
            print(prompt)
            print('-' * 20)
            print(response)
            print('#' * 50)
            print()
            
        return response
    
    def step_1_identify_targets(self, video_path, question, start, end, is_multi_choice=True):
        """
        Step 1: Task Definition and Target Identification
        
        Args:
            video_path: Path to the video file
            question: The question text
            start: Start time for video segment
            end: End time for video segment
            is_multi_choice: Whether the question is multiple choice
            
        Returns:
            The identified targets in the video relevant to the question
        """
        if is_multi_choice:
            task_definition = "You are an expert in video analysis."
        else:
            task_definition = "You are an expert in video analysis."
        
        prompt = f"{task_definition}\n\nGiven the question: \"{question}\", what are the key objects, people, or elements in the video that need to be tracked to answer this question?\n\nProvide a concise list of the key targets."
        
        response = self._generate_response(video_path, prompt, start, end, max_new_tokens=100)
        return response
    
    def step_2_object_description(self, video_path, targets, question, start, end):
        """
        Step 2: Object Description (adapted from Object Tracking in the original paper)
        
        Args:
            video_path: Path to the video file
            targets: The identified targets from step 1
            question: The original question
            start: Start time for video segment
            end: End time for video segment
            
        Returns:
            Description of the targets throughout the video
        """
        prompt = f"Describe in detail the following elements that are relevant to answering the question \"{question}\":\n\n{targets}\n\nFocus on their appearance, movement, and interactions in the video."
        
        response = self._generate_response(video_path, prompt, start, end, max_new_tokens=150)
        return response
    
    def step_3_action_analysis(self, video_path, object_descriptions, question, start, end):
        """
        Step 3: Action Analysis
        
        Args:
            video_path: Path to the video file
            object_descriptions: The object descriptions from step 2
            question: The original question
            start: Start time for video segment
            end: End time for video segment
            
        Returns:
            Analysis of actions and implications
        """
        prompt = f"Based on the question \"{question}\" and these observations:\n\n{object_descriptions}\n\nAnalyze what actions are occurring in the video, their sequence, and their implications. Include both direct observations and reasonable inferences."
        
        response = self._generate_response(video_path, prompt, start, end, max_new_tokens=200)
        return response

    def _get_first_token_logits(self, inputs, scores):
        """
        Get the first token logits for the answer prediction.
        """
        first_token_probs = torch.nn.functional.softmax(scores[0], dim=-1)
        
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
        for batch_idx in range(scores[0].shape[0]):
            logits = [
                scores[0][batch_idx, token_ids[i]].item()
                for i in range(4)
            ]
            logits_list.append(logits)
    
        return prob_list, logits_list

    def _generate_response_and_get_first_token_logits(self, video_path, prompt, start, end, max_new_tokens=100):
        """
        Generate a response and get the first token logits.
        """
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
        
        # Use more controlled generation parameters
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True, 
            output_scores=True,
        )

        sequence_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs.sequences)
        ]
        
        decoded = self.processor.batch_decode(
            sequence_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        prob_list, logits_list = self._get_first_token_logits(inputs, outputs.scores)

        response = decoded[0]
        
        # Extract just the assistant's response, removing the prompt
        if self.show_intermediate_steps:
            print('#' * 20 + ' Generate ' + '#' * 20)
            print(prompt)
            print('-' * 20)
            print(response)
            print('#' * 50)
            print()

        return response, prob_list, logits_list
    
    def step_4_answer_scoring(self, video_path, question, choices, action_analysis, start, end):
        """
        Step 4: Answer Scoring and Ranking for multi-choice questions
        
        Args:
            video_path: Path to the video file
            question: The question text
            choices: List of answer choices
            action_analysis: The action analysis from step 3
            start: Start time for video segment
            end: End time for video segment
            
        Returns:
            Final answer with scores
        """
        # First, score each choice individually
        scores_and_rationales = []
        
        for i, choice in enumerate(choices):
            prompt = f"Question: {question}\nCandidate answer: {choice[0]}\n\nBased on the video and this analysis:\n{action_analysis}\n\nRate the likelihood of this answer being correct (1-10) and explain why."
            
            response = self._generate_response(video_path, prompt, start, end, max_new_tokens=150)
            scores_and_rationales.append(response)
        
        # Now do the final ranking and selection
        prompt = f"For the question: \"{question}\", here are the ratings for each answer choice:\n\n"
        
        for i, (choice, rationale) in enumerate(zip(choices, scores_and_rationales)):
            prompt += f"Option {i+1}: {choice[0]}\nRating: {rationale}\n\n"
        
        prompt += "Based on these ratings, which answer is most likely correct and why? Give the final answer option index (respond with a single number only)."
        
        ranking_output, prob_list, logits_list = self._generate_response_and_get_first_token_logits(
            video_path, prompt, start, end, max_new_tokens=100
        )

        # Extract the final answer index assuming the model outputs only a single integer
        answer_index = None
        try:
            match = re.search(r"\b([1-9][0-9]*)\b", ranking_output.strip())
            if match:
                idx = int(match.group(1)) - 1  # Convert to 0-based index
                if 0 <= idx < len(choices):
                    answer_index = idx
                    final_answer = choices[idx][0]
                else:
                    final_answer = None
            else:
                final_answer = None
        except Exception as e:
            print(f"Parsing error: {e}")
            final_answer = None

        return final_answer, ranking_output, scores_and_rationales, prob_list, logits_list
    
    def video_qa_reasoning(self, video_path, question, choices=None, start=0, end=None, output_intermediate_steps=False, show_intermediate_steps=False):
        """
        Complete video QA reasoning process using the Video-of-Thought approach
        
        Args:
            video_path: Path to the video file
            question: The question text
            choices: List of answer choices
            start: Start time for video segment
            end: End time for video segment
            output_intermediate_steps: Whether to output intermediate reasoning steps
            show_intermediate_steps: Whether to print intermediate steps to console
            
        Returns:
            Final answer and optionally intermediate steps
        """
        start_time = perf_counter()

        self.show_intermediate_steps = show_intermediate_steps

        is_multi_choice = (choices is not None)

        if show_intermediate_steps:
            print("Step 1: Identifying targets...")
        targets = self.step_1_identify_targets(video_path, question, start, end, is_multi_choice)

        if show_intermediate_steps:
            print("Step 2: Describing objects...")
        object_descriptions = self.step_2_object_description(video_path, targets, question, start, end)

        if show_intermediate_steps:
            print("Step 3: Analyzing actions...")
        action_analysis = self.step_3_action_analysis(video_path, object_descriptions, question, start, end)

        if show_intermediate_steps:
            print("Step 4: Scoring and ranking answers...")
        final_answer, ranking_response, scores, prob_list, logits_list = self.step_4_answer_scoring(
            video_path, question, choices, action_analysis, start, end
        )
        
        # Format the final result
        if is_multi_choice:
            # Try to extract the answer index
            answer_number = 1  # default value if no answer get extracted
            
            answer_number_match = re.search(r'(\d+)', ranking_response)
            if answer_number_match:
                answer_number = int(answer_number_match.group(1))
            else:
                print(f"No answer is matched, set to default answer: {1}")

        end_time = perf_counter()
        
        if output_intermediate_steps:
            return {
                "targets": targets,
                "object_descriptions": object_descriptions,
                "action_analysis": action_analysis,
                "scores": scores,
                "answer_index": answer_number-1,
                "answer": final_answer,
                "probs": prob_list,
                "logits": logits_list,
                "inference_time": (end_time - start_time),
            }
        else:
            return final_answer
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen-VL STAR Benchmark Inference with Video-of-Thought")

    parser.add_argument("--val_pkl", type=str, default="/data/user_data/gdhanuka/STAR_dataset/STAR_val.json", help="Path to validation .pkl file")
    parser.add_argument("--video_dir", type=str, default="/data/user_data/gdhanuka/STAR_dataset/Charades_v1_480", help="Path to video directory")
    parser.add_argument("--results_file", type=str, default="/home/gdhanuka/STAR_Benchmark/analysis/qwen2vl_vot_results.jsonl", help="Path to write model predictions")
    parser.add_argument("--final_accuracy_file", type=str, default="/home/gdhanuka/STAR_Benchmark/analysis/qwen2vl_vot_final_accuracy.txt", help="Path to write final accuracy results")
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
        shuffle=True,  # randomize for better evaluation sampling
        num_workers=4,
        pin_memory=True,
    )
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    predictor = VideoOfThoughtPredictor(num_frames=num_frames)

    # Open results file in append mode
    with open(results_file, "a") as results_f:
        with tqdm(dataloader, desc="Evaluating", dynamic_ncols=True) as pbar:
            for batch in pbar:
                data_time = batch["data_proc_time"][0]
                start = batch["start"][0].item()
                end = batch["end"][0].item()
                video_frames = batch["video_frames"][0].to(device)
                question = batch["question"][0]
                choices = batch["choices"]
                answer_idx = batch["answer_idx"][0]
                category = batch["category"][0]
                all_text_inputs = batch["all_text_inputs"][0]
                question_id = batch["question_id"][0]
                frame_ids = batch["frame_ids"][0]
                video_path = batch["video_path"][0]

                # Get model prediction
                result = predictor.video_qa_reasoning(
                    video_path=video_path,
                    question=question,
                    choices=choices,
                    start=start,
                    end=end,
                    show_intermediate_steps=False,
                    output_intermediate_steps=True
                )
                
                answer = result["answer_index"]
                final_answer_text = result["answer"]
                probs = result["probs"]
                logits = result["logits"]
                targets = result["targets"]
                object_descriptions = result["object_descriptions"]
                action_analysis = result["action_analysis"]
                scores = result["scores"]
                inference_time = result["inference_time"]

                # Save result to JSONL file
                json_record = {
                    "question_id": question_id,
                    "question": question,
                    "choices": [c[0] for c in choices],
                    "pred_ans_idx": answer,
                    "true_index": (answer_idx).item(),
                    "category": category,
                    "raw_response": final_answer_text,
                    "prompt": "",  # You can fill this if you want to log full prompts
                    "frame_ids": frame_ids.numpy().tolist(),
                    "inference_time": inference_time,
                    "data_time": data_time.item(),
                    "confidence": probs[0] if probs else [],
                    "logits": logits[0] if logits else [],
                    "targets": targets,
                    "object_descriptions": object_descriptions,
                    "action_analysis": action_analysis,
                    "scores": scores,
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

python baselines/qwen2vl_vot.py \
    --val_pkl "/data/user_data/jamesdin/STAR/data/STAR_val.json" \
    --video_dir "/data/user_data/jamesdin/STAR/data/Charades_v1_480" \
    --results_file "analysis/results/qwen2vl_vot_results.jsonl" \
    --final_accuracy_file "analysis/results/qwen2vl_vot_final_accuracy.txt" \
    --num_frames 8
"""