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
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from huggingface_hub import hf_hub_download

from video_qa_dataset import VideoQADataset

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

    def generate(self, inputs, max_new_tokens=500):
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            return_dict_in_generate=True, output_scores=True
        )
        decoded = self.processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
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


        return decoded, prob_list, logits_list

    def video_qa(self, video_frames, question, choices, max_new_tokens=500):
        choice_with_idx = [f"{i+1}: {choice[0]}" for i, choice in enumerate(choices)]
        choice_str = "\n".join(choice_with_idx)
        prompt = f"USER: <video>\n {question} \n {choice_str} \n Answer with only the option's index from the given choices directly. \n ASSISTANT: "
        # print(prompt)
        # choice_with_idx = [f'"{i+1}": {choice}\n' for i, choice in enumerate(choices)]
        # prompt = f"USER: <video>\n {question} \n {choice_with_idx} Answer with the option's index from the given choices directly. \n ASSISTANT: "
        inputs = self.processor(
            text=prompt, videos=video_frames, return_tensors="pt", max_length=4096
        ).to("cuda")
        decoded, probs, logits = self.generate(inputs, max_new_tokens=max_new_tokens)
        return decoded[0], probs[0], prompt, logits[0]

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
    dataset = VideoQADataset(val_pkl, video_dir=video_dir, sampling_fps=4, num_frames=8, use_fps=False)
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

    video_qa_model = VideoQAModel(load_in_bits=16)  # set to load_in_bits=4 to save GPU memory

    # Open results file in append mode
    with open(results_file, "a") as results_f:
        with tqdm(dataloader, desc="Evaluating", dynamic_ncols=True) as pbar:
            for batch in pbar:
                start_time = perf_counter()
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
                answer, probs, prompt, logits = video_qa_model.video_qa(video_frames, question, choices)
                answer_raw = answer
                answer = int(re.search(r"\d+", answer.split("ASSISTANT")[-1]).group())
                end_time = perf_counter()

                # Save result to JSONL file
                json_record = {
                    "question_id": question_id,
                    "question": question,
                    "choices": choices,
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


python baselines/video-llava.py \
    --val_pkl "/data/user_data/jamesdin/STAR/data/STAR_val.pkl" \
    --video_dir "/data/user_data/jamesdin/STAR/data/Charades_v1_480" \
    --results_file "analysis/video_llava_4_frames_results.jsonl" \
    --final_accuracy_file "analysis/video_llava_4_frames_final_accuracy.txt" \
    --load_in_bits 4 \
    --num_frames 8


"""
