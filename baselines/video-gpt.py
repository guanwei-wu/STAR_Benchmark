import base64
import json
import os
import re
from collections import defaultdict
from io import BytesIO
from time import perf_counter
import argparse

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from openai import OpenAI
from video_llava import VideoQADataset

class GPTImageQAEvaluator:
    def __init__(self, api_key, model_name="gpt-4o", base_url="https://cmu.litellm.ai"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def _encode_tensor_to_base64(self, image_tensor: torch.Tensor) -> str:
        # image_tensor: (C, H, W), value range: [0, 255]
        image_np = image_tensor.to(torch.uint8).permute(1, 2, 0).cpu().numpy()  # to (H, W, C)
        pil_img = Image.fromarray(image_np)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def video_qa(self, video_frames: torch.Tensor, question: str, choices: list[str]):
        # video_frames: (T, C, H, W)
        image_sections = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._encode_tensor_to_base64(frame)}"}
            }
            for frame in video_frames
        ]
        breakpoint()
        # Generate prompt with questions and choices
        formatted_choices = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
        prompt = f"USER: <video>\n{question}\n{formatted_choices}\nAnswer with the option's index from the given choices directly.\nASSISTANT:"
        print(prompt)

        messages = [  # create complete prompt content
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + image_sections
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            logprobs=True,
            top_logprobs=4,  # return top 4 logprobs
        )

        raw_answer = response.choices[0].message.content.strip()
        print(f"Raw answer from {self.model_name}: {raw_answer}")
        match = re.search(r"\d+", raw_answer)
        if match:
            pred_idx = int(match.group()) - 1  # convert to 0-based index
        else:
            pred_idx = -1  # error

        # Calculate the confidence (probability) for all the options
        logprobs_list = response.choices[0].logprobs.content
        logits_dict = {}
        if logprobs_list and hasattr(logprobs_list[0], "top_logprobs"):
            for item in logprobs_list[0].top_logprobs:
                token = item.token.strip()
                logits_dict[token] = item.logprob
        
        # Normalize the logits to get the confidence
        tokens = list(logits_dict.keys())
        logits = torch.tensor([logits_dict[token] for token in tokens])
        probs = torch.softmax(logits, dim=0)
        confidence_dict = {token: prob.item() for token, prob in zip(tokens, probs)}

        return raw_answer, pred_idx, prompt, logits_dict, confidence_dict


def show_video_frames(video_frames, question=None):
    """
    video_frames: Tensor of shape (T, C, H, W)
    """
    num_frames = video_frames.shape[0]
    fig, axs = plt.subplots(1, num_frames, figsize=(3 * num_frames, 3))

    for i in range(num_frames):
        frame = video_frames[i]

        if frame.dtype == torch.uint8:
            frame = frame.float() / 255.0
        elif frame.max() > 1.0:
            frame = frame / 255.0

        pil_image = TF.to_pil_image(frame)
        axs[i].imshow(pil_image)
        axs[i].axis('off')
        axs[i].set_title(f"Frame {i}")

    if question:
        plt.suptitle(question, fontsize=14)

    plt.tight_layout()
    plt.show()


def evaluate_model(args):
    print("Start evaluating the model on the validation set...")
    api_key = os.environ.get("LITELLM_API_KEY")
    evaluator = GPTImageQAEvaluator(api_key=api_key, model_name=args.model)

    dataset = VideoQADataset(
        args.val_pkl,
        video_dir=args.video_dir,
        sampling_fps=4,
        num_frames=8,
        use_fps=False,
    )
    print(f"Finish loading dataset. Total number of samples in the validation set: {len(dataset)}")

    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

    with open(args.results_file, "w") as results_f:
        for sample in tqdm(dataset, desc="Evaluating", dynamic_ncols=True):
            start_time = perf_counter()

            try:
                raw_answer, pred_idx, prompt, logits_dict, confidence_dict = evaluator.video_qa(
                    sample["video_frames"], sample["question"], sample["choices"]
                )
            except Exception as e:
                print(f"[ERROR] QID={sample['question_id']} failed: {e}")
                raw_answer = "error"
                pred_idx = -1

            end_time = perf_counter()

            record = {
                "question_id": sample["question_id"],
                "question": sample["question"],
                "choices": sample["choices"],
                "pred_ans_idx": pred_idx,
                "true_index": sample["answer_idx"],
                "category": sample["category"],
                "raw_response": raw_answer,
                "prompt": prompt,
                "frame_ids": sample["frame_ids"].tolist(),
                "inference_time": end_time - start_time,
                "data_time": sample["data_proc_time"],
                "confidence": confidence_dict,
                "logits": logits_dict,
            }

            results_f.write(json.dumps(record) + "\n")

            cat = sample["category"]
            category_total[cat] += 1
            if pred_idx == sample["answer_idx"]:
                category_correct[cat] += 1

            acc_info = {c: f"{category_correct[c] / max(category_total[c], 1):.3f}" for c in category_total}
            acc_info["Overall"] = f"{sum(category_correct.values()) / max(sum(category_total.values()), 1):.3f}"
            tqdm.write(json.dumps(acc_info))

    with open(args.accuracy_file, "w") as acc_f:
        acc_f.write("Final Category-Wise Accuracy:\n")
        for cat in category_total:
            acc_f.write(f"{cat}: {category_correct[cat] / category_total[cat]:.4f}\n")
        acc_f.write(f"\nOverall accuracy: {sum(category_correct.values()) / sum(category_total.values()):.4f}\n")

    print(f"Results saved to {args.results_file} and accuracy to {args.accuracy_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run GPT-based video QA inference")
    parser.add_argument("--val_pkl", type=str, default="/home/rickylinux/dataset/MMML/STAR_val.pkl", help="Path to validation pkl file")
    parser.add_argument("--video_dir", type=str, default="/home/rickylinux/dataset/MMML/Charades_v1_480", help="Path to video directory")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", choices=["gpt-4o", "gpt-4o-mini"], help="Model name to use")
    parser.add_argument("--results_file", type=str, default=f"analysis/video_gpt_results.jsonl")
    parser.add_argument("--accuracy_file", type=str, default=f"analysis/video_gpt_accuracy.txt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)