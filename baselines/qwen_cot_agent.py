# # In your code, add these lines at the top of your script (before other imports)
# import debugpy

# # Set up the debugger to listen on all interfaces
# debugpy.listen(("0.0.0.0", 5678))  # Use a port number that's available
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()  # This line will pause execution until VSCode connects
# print("Debugger attached!")

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
import gc
from time import perf_counter

# warnings.filterwarnings("ignore")

from IPython.display import HTML
import os
import pickle
from sentence_transformers import SentenceTransformer
import av

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
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from huggingface_hub import hf_hub_download

# from video_qa_dataset import VideoQADataset
from qwen_vl_utils import process_vision_info

from huggingface_hub import hf_hub_download


import re
import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList



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
    
# def process_vision_info(messages):
#     """Extract image and video inputs from messages."""
#     image_inputs = []
#     video_inputs = []

#     # Process messages to extract image and video data
#     for message in messages:
#         if message["role"] == "user":
#             for content_item in message["content"]:
#                 # Extract video content - using the exact format from your original code
#                 if content_item.get("type") == "video":
#                     # Pass the video object directly as is - don't reformat it
#                     # This preserves the exact structure expected by your processor
#                     video_path = content_item["video"]
#                     start = content_item.get("video_start", 0.0)
#                     end = content_item.get("video_end", None)
#                     nframes = content_item.get("nframes", 8)

#                     # Your processor might expect the video directly, not in a dictionary
#                     video_inputs.append(content_item)

#                 # Extract image content
#                 elif content_item.get("type") == "image":
#                     # Similarly, preserve the original format for images
#                     image_inputs.append(content_item)

#         # Also check assistant messages for retrieved frames (which are images)
#         elif message["role"] == "assistant":
#             for content_item in message["content"]:
#                 if isinstance(content_item, dict) and content_item.get("type") == "image":
#                     image_inputs.append(content_item)

#     print(f"Extracted: {len(video_inputs)} video inputs, {len(image_inputs)} image inputs")
#     return image_inputs, video_inputs

def retrieve_frames(query: str, video_path: str, start: float, end: float) -> List[Dict]:
    """
    Retrieve relevant frames based on the query.
    This is a placeholder for your actual retrieval logic.
    
    Args:
        query: The text query to search for in the video
        video_path: Path to the video file
        start: Start time in the video
        end: End time in the video
        
    Returns:
        List of retrieved frames in the format needed by the model
    """
    # Implement your frame retrieval logic here
    # This could be a call to a separate model or system
    print(f"Retrieving frames for query: {query}")
    
    # In a real implementation, this function would:
    # 1. Process the query to identify relevant time points in the video
    # 2. Extract frames from those time points
    # 3. Return the frames in the format expected by your model
    
    # For demonstration, let's assume we've retrieved 3 frames from different times
    # This is a placeholder - you'll replace this with your actual retrieval logic
    
    image_path = "/data/user_data/jamesdin/test_data/001.jpg"  # dummy image
    messages =  [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                ],
            }
    ]
    retrieved_frames, _ = process_vision_info(messages)
    
    # IMPORTANT: Make sure the format matches exactly what your model expects
    # This should match the format in your original code's user message
    # return [
    #     {
    #         "type": "image",
    #         "image": f"{video_path}_retrieved_frame_{i}.jpg",  # Path to retrieved frame
    #         "caption": f"Retrieved frame for query: {query}"
    #     }
    #     for i in range(3)  # Returning 3 sample frames
    # ]
    return retrieved_frames

def interactive_video_qa(
    video_qa_model,
    processor,
    system_prompt: str,
    user_prompt: str,
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

    # Format the user prompt with the question and choices
    formatted_user_prompt = user_prompt.format(
        question=question,
        choices='\n'.join([f"{i+1}. {c}" for i, c in enumerate(choices)])
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

    # Start with empty assistant response
    # assistant_content = []
    full_response = ""
    
    messages = initial_messages.copy()

    # Prepare inputs for the model
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Extract the vision info but pass the original messages to the processor
    # This ensures the processor handles the video/image data in its expected format
    image_inputs, video_inputs = process_vision_info(messages)
    # print(f"[Iteration {iteration}] Processing with {len(video_inputs)} videos and {len(image_inputs) if image_inputs else 0} images")
    

    # Perform iterative generation with look-retrieve cycles
    for iteration in range(max_iterations):
        # Reset messages to initial state each iteration
        # This ensures we always have the video in the context

        # If we have assistant content from previous iterations, add it
        # if assistant_content:
        #     messages.append({
        #         "role": "assistant",
        #         "content": assistant_content
        #     })

        # Pass the entire messages object to the processor instead of extracted components
        # This is crucial if your processor expects a specific format for videos/images
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Generate continuation
        if iteration > 0 and "</look>" in full_response:
            # Regular generation for continuation after look tags
            outputs = video_qa_model.generate(
                **inputs,
                max_new_tokens=100,
                return_dict_in_generate=True,
                output_scores=True,
            )
        else:
            # First iteration or when we want to stop at look tags
            # Use a custom stopping criteria at </look> token
            # outputs = video_qa_model.generate(
            #     **inputs,
            #     max_new_tokens=100,
            #     # eos_token_id=look_end_token_id,  # Stop at </look> token
            #     return_dict_in_generate=True,
            #     output_scores=True,
            # )
            
            input_length = len(inputs.input_ids[0])  # Get length of input context
            look_stopping_criteria = LookEndTokenStoppingCriteria(
                processor.tokenizer,
                input_length=input_length  # Pass input length to ignore
            )
                        
            outputs = video_qa_model.generate(
                **inputs,
                max_new_tokens=100,
                stopping_criteria=StoppingCriteriaList([look_stopping_criteria]),
                # return_dict_in_generate=True,
                # output_scores=True,
            )


        # Decode the generated text
        # generated_ids = outputs.sequences[0]
        generated_ids = outputs[0]
        generated_text = processor.tokenizer.decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Extract just the new content (avoid repeating previous content)
        # This might need adjustment based on your specific model output format
        new_content_start = len(text) if iteration == 0 else len(text) + len(full_response)
        new_content = generated_text[new_content_start:]

        # Check if we have a look query
        look_match = re.search(r"<look>(.*?)</look>", new_content)
        if look_match:
            # Extract the query and everything before it
            query = look_match.group(1).strip()
            before_look = new_content[:look_match.start()]

            # Update full response with content up to the look tag and include the tag itself
            # This ensures the model's context includes the previous query
            full_response += before_look + f"<look>{query}</look>\n"
            # full_response += new_content

            # Retrieve frames based on the query
            retrieved_frames = retrieve_frames(query, video_path, start, end)

            # Add the retrieved frames to the assistant content
            # The text response should include everything generated so far
            # assistant_content = [
            #     {"type": "text", "text": full_response}
            # ] + retrieved_frames
            
            # for each frame in retrieved_frames, we add a "<|vision_start|><|image_pad|><|vision_end|>" block
            for frame in retrieved_frames:
                text += f"<|vision_start|>{frame['image']}<|image_pad|><|vision_end|>"
            
            # add the image file to the image_inputs
            if image_inputs is None:
                image_inputs = retrieved_frames
            else:
                image_inputs.extend(retrieved_frames)

            # Important: Print for debugging
            print(f"[Iteration {iteration}] Retrieved frames for query: {query}")
            print(f"[Iteration {iteration}] Current response: {full_response}")
            print(f"[Iteration {iteration}] Retrieved frames for query: {query}")

            print(f"Iteration {iteration+1}: Retrieved frames for query '{query}'")
        else:
            # No more look queries, finalize the response
            full_response += new_content
            print(f"Iteration {iteration+1}: Generation complete")
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

# --- Construct Prompt ---
system_prompt = "You are a helpful video reasoning assistant."
user_prompt = f"""
You are a video reasoning assistant. When answering questions about video content:

- Use numbered steps to reason through the problem.
- When you need visual information, write a single line with: <look> your query </look>
  (This will retrieve frames from the video for you to use in your next step.)
- End with: Answer: <choice number>. <choice text>

Follow the examples below:

--------------------------------------------------
Example 1
Question: What does the chef do *after* adding salt?
Choices:
1. Tastes the soup.
2. Adds pepper.
3. Turns off the stove.
4. Puts on the lid.

Step 1: I need to see what the chef does right after adding salt.
<look> chef’s hands immediately after salt is poured </look>
Step 2: The chef grabs a pepper shaker and adds pepper to the pot.
Step 3: So the action that follows adding salt is adding pepper.
Answer: 2. Adds pepper.
--------------------------------------------------
Example 2
Question: Why does the girl scream?
Choices:
1. She sees a mouse.
2. She drops her ice cream.
3. Someone surprises her.
4. She wins a prize.

Step 1: First, check the scene right before the girl screams.
<look> frames right before the girl screams </look>
Step 2: A person jumps out from behind the door.
Step 3: So the girl screams because she was startled.
Answer: 3. Someone surprises her.
--------------------------------------------------

Now answer the following:

Question: {question}
Choices:
{''.join([f"{i+1}. {choice}" + chr(10) for i, choice in enumerate(choices)])}
"""

# --- Run Model ---
result = interactive_video_qa(
    model,
    processor,
    system_prompt,
    user_prompt,
    video_path,
    start,
    end,
    question,
    choices
)

print("Final response:")
print(result)
