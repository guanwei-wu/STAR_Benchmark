import argparse
import gc
import json
import os
import pickle
import re
from collections import defaultdict
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import av
import numpy as np
from IPython.display import HTML
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adagrad
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from huggingface_hub import hf_hub_download

from sentence_transformers import SentenceTransformer
from qwen_vl_utils import process_vision_info
from video_qa_dataset import VideoQADataset

from frame_retriever import retrieve_frames

class LookEndTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, input_length, stop_string="</look>"):
        self.tokenizer = tokenizer
        self.stop_string = stop_string
        self.input_length = input_length
        
    def __call__(self, input_ids, scores, **kwargs):
        # For each sequence in the batch
        for input_id in input_ids:
            # Only look at newly generated tokens
            if len(input_id) <= self.input_length:
                continue
                
            # Decode the generated text (keeping special tokens)
            generated_text = self.tokenizer.decode(
                input_id[self.input_length:], 
                skip_special_tokens=False
            )
            
            # Simple string match for the stop string
            if self.stop_string in generated_text:
                return True
        return False

def interactive_video_qa(
    video_qa_model,
    processor,
    video_path: str,
    start: float,
    end: float,
    question: str,
    choices: List[str],
    max_iterations: int = 5,
    device: str = "cuda"
) -> str:
    """
    Run an interactive video QA pipeline with look-and-retrieve functionality.
    
    Args:
        video_qa_model: The main language model for reasoning
        processor: Processor for the model
        system_prompt: System prompt
        user_prompt: User prompt template
        video_path: Path to the video file
        start: Start time in the video
        end: End time in the video
        question: The question to answer
        choices: List of possible answers
        max_iterations: Maximum number of look-retrieve cycles
        device: Device to run inference on
        
    Returns:
        The complete generated response
    """

    # Do not add this section for the zero-shot evaluation, because the model didn't not learn to generate this token yet
    # Add this part before the training.
    # Add special tokens to the tokenizer
    # special_tokens = {'additional_special_tokens': ['<look>', '</look>']}
    # processor.tokenizer.add_special_tokens(special_tokens)

    # # Resize the model's token embeddings to accommodate the new tokens
    # model.resize_token_embeddings(len(processor.tokenizer))

    # # Get the token IDs for later use
    # look_token_id = processor.tokenizer.convert_tokens_to_ids('<look>')
    # look_end_token_id = processor.tokenizer.convert_tokens_to_ids('</look>')

    # print(f"New tokens added: <look>: {look_token_id}, </look>: {look_end_token_id}")
    
    from prompts import system_prompt, user_prompt

    # Format the user prompt with the question and choices
    formatted_user_prompt = user_prompt.format(
        question=question,
        choices=''.join([f"{i+1}. {c}" + chr(10) for i, c in enumerate(choices)])
    )

    # Initial messages with video
    initial_messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "video_start": start,
                    "video_end": end,
                    "nframes": 8
                },
                {"type": "text", "text": formatted_user_prompt},
            ],
        }
    ]

    # Prepare the initial template only once
    base_text = processor.apply_chat_template(initial_messages, tokenize=False, add_generation_prompt=True)
    current_text = base_text  # Text to be used for this iteration
    
    # Initialize variables
    full_response = ""  # The accumulated assistant's response
    image_inputs = None   # Will hold retrieved frames
    
    # Extract initial video inputs
    _, video_inputs = process_vision_info(initial_messages)
    
    print(f"[Start] Starting generation, max iterations: {max_iterations}")

    # Perform iterative generation with look-retrieve cycles
    for iteration in range(max_iterations):
        
        # Prepare inputs for the processor
        inputs = processor(
            text=[current_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Set up stopping criteria to find </look> tokens in newly generated content
        input_length = len(inputs.input_ids[0])
        look_stopping_criteria = LookEndTokenStoppingCriteria(
            processor.tokenizer,
            input_length=input_length,
            stop_string="</look>"
        )
        
        # Generate text with stopping at </look>
        outputs = video_qa_model.generate(
            **inputs,
            max_new_tokens=100,
            stopping_criteria=StoppingCriteriaList([look_stopping_criteria]),
        )

        # Decode the generated text
        input_length = len(inputs.input_ids[0])
        generated_ids = outputs[0]
        new_token_ids = generated_ids[input_length:]

        # Decode only the new tokens to text
        new_content = processor.tokenizer.decode(
            new_token_ids, 
            skip_special_tokens=False, 
            clean_up_tokenization_spaces=False
        )
        
        print(f"[Iteration {iteration}] New content: {new_content}")

        # Check if we have a look query
        look_match = re.search(r"<look>(.*?)</look>", new_content)
        if look_match:
            # Extract the query
            query = look_match.group(1).strip()
            # before_look = new_content[:look_match.end()]  # Include the </look> tag
            # # before_look = new_content[:look_match.start()]
            
            # # Update full response
            full_response += new_content
            
            print(f"[Iteration {iteration}] Retrieved frames for query: {query}")
            
            # Retrieve frames based on the query
            retrieved_frames = retrieve_frames(query, video_path, start, end)
            
            # Add the retrieved frames to image_inputs
            if not image_inputs:
                image_inputs = retrieved_frames
            else:
                image_inputs.extend(retrieved_frames)
            
            # Update the current text for next iteration
            # We need to include: original prompt + full response so far + vision markers
            current_text = base_text + full_response
            
            # Add vision markers for each new frame in the format Qwen expects
            for frame in retrieved_frames:
                current_text += "\n<|vision_start|><|image_pad|><|vision_end|>"
                new_content += str(frame)  # can print out the actual retrieved frames for visualization
                
                print(f"[Iteration {iteration}] Retrieved frames: {str(frame)}")
            
        else:
            # No more look queries, finalize the response
            full_response += new_content
            break

    return full_response

import pickle
import pandas as pd

# --- Load Model ---
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


# --- Load Data ---
# Load the .pkl file
with open('/data/user_data/jamesdin/STAR/data/STAR_val.pkl', 'rb') as f:
    data = pickle.load(f)
# Convert to DataFrame and set index
df = pd.DataFrame(data).set_index('question_id')

question_id = 'Sequence_T1_6700'
example = df.loc[question_id]
video_id = example['video_id']
video_path = f"/data/user_data/jamesdin/STAR/data/Charades_v1_480/{video_id}.mp4"
question = example['question']
choices = [x['choice'] for x in example['choices']]
start = example['start']
end = example['end']


# --- Run Model ---
result = interactive_video_qa(
    model,
    processor,
    video_path,
    start,
    end,
    question,
    choices
)

print("Final response:")
print(result)
