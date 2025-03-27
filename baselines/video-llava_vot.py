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
import argparse

from video_qa_dataset import VideoQADataset

import torch
import re
from typing import List, Dict, Any, Optional, Union
import numpy as np
from time import perf_counter
import json
from tqdm import tqdm
import os


class VideoOfThoughtPredictor:
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
                # load_in_4bit=load_in_bits == 4,
                # load_in_8bit=load_in_bits == 8,
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
        else:
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                "LanguageBind/Video-LLaVA-7B-hf",
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            ).to("cuda")
        self.processor = VideoLlavaProcessor.from_pretrained(
            "LanguageBind/Video-LLaVA-7B-hf"
        )
        
        self.show_intermediate_steps = False
        
    def _generate_response(self, video_frames, prompt, max_new_tokens=100):
        """
        Generate a response from the VideoLLAVA model.
        
        Args:
            video_frames: Video frame tensors
            prompt: Text prompt
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        inputs = self.processor(
            text=prompt, 
            videos=video_frames, 
            return_tensors="pt", 
            max_length=4096
        ).to("cuda")
        
        # Use more controlled generation parameters
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # do_sample=False,  # Use greedy decoding for more consistent outputs
            # temperature=0.1,   # Lower temperature for more focused responses
            # num_beams=1,
            # early_stopping=True,
            # pad_token_id=self.model.processor.tokenizer.pad_token_id,
            # eos_token_id=self.model.processor.tokenizer.eos_token_id
        )

        
        decoded = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        response = decoded[0]
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        # Extract just the assistant's response, removing the prompt
        if self.show_intermediate_steps:
            print('#' * 50)
            print(prompt)
            print('-' * 20)
            print("ASSISTANT:" + response)
            print('#' * 50)
            print()
            
        return response
    
    def step_1_identify_targets(self, video_frames, question, is_multi_choice=True):
        """
        Step 1: Task Definition and Target Identification
        
        Args:
            video_frames: Video frame tensors
            question: The question text
            is_multi_choice: Whether the question is multiple choice
            
        Returns:
            The identified targets in the video relevant to the question
        """
        if is_multi_choice:
            task_definition = "You are an expert in video analysis."
        else:
            task_definition = "You are an expert in video analysis."
        
        prompt = f"USER: <video>\n{task_definition}\n\nGiven the question: \"{question}\", what are the key objects, people, or elements in the video that need to be tracked to answer this question?\n\nProvide a concise list of the key targets.\nASSISTANT:"
        
        response = self._generate_response(video_frames, prompt, max_new_tokens=100)
        return response
    
    def step_2_object_description(self, video_frames, targets, question):
        """
        Step 2: Object Description (adapted from Object Tracking in the original paper)
        
        Args:
            video_frames: Video frame tensors
            targets: The identified targets from step 1
            question: The original question
            
        Returns:
            Description of the targets throughout the video
        """
        prompt = f"USER: <video>\nDescribe in detail the following elements that are relevant to answering the question \"{question}\":\n\n{targets}\n\nFocus on their appearance, movement, and interactions in the video.\nASSISTANT:"
        
        response = self._generate_response(video_frames, prompt, max_new_tokens=150)
        return response
    
    def step_3_action_analysis(self, video_frames, object_descriptions, question):
        """
        Step 3: Action Analysis
        
        Args:
            video_frames: Video frame tensors
            object_descriptions: The object descriptions from step 2
            question: The original question
            
        Returns:
            Analysis of actions and implications
        """
        prompt = f"USER: <video>\nBased on the question \"{question}\" and these observations:\n\n{object_descriptions}\n\nAnalyze what actions are occurring in the video, their sequence, and their implications. Include both direct observations and reasonable inferences.\nASSISTANT:"
        
        response = self._generate_response(video_frames, prompt, max_new_tokens=200)
        return response


    def _get_first_token_logits(self, scores):

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

    def _generate_response_and_get_first_token_logits(self, video_frames, prompt, max_new_tokens=100):
        
        inputs = self.processor(
            text=prompt, 
            videos=video_frames, 
            return_tensors="pt", 
            max_length=4096
        ).to("cuda")
        
        # Use more controlled generation parameters
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True, 
            output_scores=True,
            # do_sample=False,  # Use greedy decoding for more consistent outputs
            # temperature=0.1,   # Lower temperature for more focused responses
            # num_beams=1,
            # early_stopping=True,
            # pad_token_id=self.model.processor.tokenizer.pad_token_id,
            # eos_token_id=self.model.processor.tokenizer.eos_token_id
        )

        
        decoded = self.processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        prob_list, logits_list = self._get_first_token_logits(outputs.scores)

        response = decoded[0]
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        # Extract just the assistant's response, removing the prompt
        if self.show_intermediate_steps:
            print('#' * 20 + ' Generate ' + '#' * 20)
            print(prompt)
            print('-' * 20)
            print("ASSISTANT:" + response)
            print('#' * 50)
            print()

        return response, prob_list, logits_list
            
    
    def step_4_answer_scoring(self, video_frames, question, choices, action_analysis):
        """
        Step 4: Answer Scoring and Ranking for multi-choice questions
        
        Args:
            video_frames: Video frame tensors
            question: The question text
            choices: List of answer choices
            action_analysis: The action analysis from step 3
            
        Returns:
            Final answer with scores
        """
        # First, score each choice individually
        scores_and_rationales = []
        
        for i, choice in enumerate(choices):
            prompt = f"USER: <video>\nQuestion: {question}\nCandidate answer: {choice}\n\nBased on the video and this analysis:\n{action_analysis}\n\nRate the likelihood of this answer being correct (1-10) and explain why.\nASSISTANT:"
            
            response = self._generate_response(video_frames, prompt, max_new_tokens=150)
            scores_and_rationales.append(response)
        
        # Now do the final ranking and selection
        prompt = f"USER: <video>\nFor the question: \"{question}\", here are the ratings for each answer choice:\n\n"
        
        for i, (choice, rationale) in enumerate(zip(choices, scores_and_rationales)):
            prompt += f"Option {i+1}: {choice}\nRating: {rationale}\n\n"
        
        prompt += "Based on these ratings, which answer is most likely correct and why? Give the final answer option index (respond with a single number only).\nASSISTANT:"
        
        ranking_output, prob_list, logits_list = self._generate_response_and_get_first_token_logits(video_frames, prompt, max_new_tokens=100)

        # Extract the final answer index assuming the model outputs only a single integer
        answer_index = None
        try:
            match = re.search(r"\b([1-9][0-9]*)\b", ranking_output.strip())
            if match:
                idx = int(match.group(1)) - 1  # Convert to 0-based index
                if 0 <= idx < len(choices):
                    answer_index = idx
                    final_answer = choices[idx]
                else:
                    final_answer = None
            else:
                final_answer = None
        except Exception as e:
            print(f"Parsing error: {e}")
            final_answer = None

        return final_answer, ranking_output, scores_and_rationales, prob_list, logits_list
    
    def step_5_answer_verification(self, video_frames, question, final_answer, action_analysis):
        """
        Step 5: Answer Verification
        
        Args:
            video_frames: Video frame tensors
            question: The question text
            final_answer: The final answer from step 4
            action_analysis: The action analysis from step 3
            
        Returns:
            Verification of the answer
        """
        prompt = f"USER: <video>\nQuestion: {question}\nSelected answer: {final_answer}\n\nBased on the video evidence and this analysis:\n{action_analysis}\n\nVerify whether this answer is correct. Provide a final verdict (correct/incorrect) with justification.\nASSISTANT:"
        
        response = self._generate_response(video_frames, prompt, max_new_tokens=150)
        
        return response
    
    def video_qa_reasoning(self, video_frames, question, choices=None, output_intermediate_steps=False, show_intermediate_steps=False):
        """
        Complete video QA reasoning process using the Video-of-Thought approach
        
        Args:
            video_frames: Video frame tensors
            question: The question text
            choices: List of answer choices
            output_intermediate_steps: Whether to output intermediate reasoning steps
            
        Returns:
            Final answer and optionally intermediate steps
        """

        start_time = perf_counter()

        self.show_intermediate_steps = show_intermediate_steps

        is_multi_choice = (choices is not None)

        if show_intermediate_steps:
            print("Step 1: Identifying targets...")
        targets = self.step_1_identify_targets(video_frames, question, is_multi_choice)

        if show_intermediate_steps:
            print("Step 2: Describing objects...")
        object_descriptions = self.step_2_object_description(video_frames, targets, question)

        if show_intermediate_steps:
            print("Step 3: Analyzing actions...")
        action_analysis = self.step_3_action_analysis(video_frames, object_descriptions, question)

        if show_intermediate_steps:
            print("Step 4: Scoring and ranking answers...")
        final_answer, ranking_response, scores, prob_list, logits_list = self.step_4_answer_scoring(
            video_frames, question, choices, action_analysis
        )

        # if show_intermediate_steps:
        #     print("Step 5: Verifying answer...")
        # verification = self.step_5_answer_verification(
        #     video_frames, question, final_answer, action_analysis
        # )
        
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
                "probs": prob_list,  # confidence for each option choice from the final reasoning step
                "logits": logits_list,  # raw logits of each option choice from the final reasoning step
                "inference_time": (end_time - start_time),
                # "verification": verification,
                # "final_result": final_result
            }
        else:
            return final_answer
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoLLAVA STAR Benchmark Inference")

    parser.add_argument("--val_pkl", type=str, default="data/STAR_val.pkl", help="Path to validation .pkl file")
    parser.add_argument("--video_dir", type=str, default="data/Charades_v1_480", help="Path to video directory")
    parser.add_argument("--results_file", type=str, default="analysis/video_llava_4_frames_results.jsonl", help="Path to write model predictions")
    parser.add_argument("--final_accuracy_file", type=str, default="analysis/video_llava_4_frames_final_accuracy.txt", help="Path to write final accuracy results")
    parser.add_argument("--load_in_bits", type=int, default=16, choices=[4, 8, 16], help="Precision for model loading (4, 8, or 16)")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of video frames to sample for inference")

    args = parser.parse_args()

    # Now use these variables below
    val_pkl = args.val_pkl
    video_dir = args.video_dir
    results_file = args.results_file
    final_accuracy_file = args.final_accuracy_file
    load_in_bits = args.load_in_bits
    num_frames = args.num_frames

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = VideoQADataset(val_pkl, video_dir=video_dir, sampling_fps=4, num_frames=num_frames, use_fps=False)
    # batched inference not working!!
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,  # disable to easily reproduce
        num_workers=4,
        pin_memory=True,
        # collate_fn=collate_fn,
    )
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    # video_qa_model = VideoQAModel(load_in_bits=load_in_bits)  # set to load_in_bits=4 to save GPU memory
    predictor = VideoOfThoughtPredictor(load_in_bits=load_in_bits)

    # Open results file in append mode
    with open(results_file, "a") as results_f:
        with tqdm(dataloader, desc="Evaluating", dynamic_ncols=True) as pbar:
            for batch in pbar:
                
                data_time = batch["data_proc_time"][0]
                video_frames = batch["video_frames"][0].to(device)
                question = batch["question"][0]
                choices = batch["choices"]
                answer_idx = batch["answer_idx"][0]
                category = batch["category"][0]
                all_text_inputs = batch["all_text_inputs"][0]
                question_id = batch["question_id"][0]
                frame_ids = batch["frame_ids"][0]

                # Get model prediction
                result = predictor.video_qa_reasoning(
                    video_frames=video_frames,
                    question=question,
                    choices=choices,
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
                    "choices": choices,
                    "pred_ans_idx": answer,
                    "true_index": (answer_idx).item(),
                    "category": category,
                    "raw_response": final_answer_text,
                    "prompt": "",  # You can fill this if you want to log full prompts
                    "frame_ids": frame_ids.numpy().tolist(),
                    "inference_time": inference_time,
                    "data_time": data_time.item(),
                    "confidence": probs,
                    "logits": logits,
                    "targets": targets,
                    "object_descriptions": object_descriptions,
                    "action_analysis": action_analysis,
                    "scores": scores,
                }

                # print type of each item in json_record
                # for key, val in json_record.items():
                #     print(f"{key}: {type(val)}")
                # print('answer',answer)

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


"""


python baselines/video-llava_vot.py \
    --val_pkl "/data/user_data/jamesdin/STAR/data/STAR_val.pkl" \
    --video_dir "/data/user_data/jamesdin/STAR/data/Charades_v1_480" \
    --results_file "analysis/video_llava_vot_results.jsonl" \
    --final_accuracy_file "analysis/video_llava_vot_final_accuracy.txt" \
    --load_in_bits 16 \
    --num_frames 8


"""