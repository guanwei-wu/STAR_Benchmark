import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image
from IPython.display import HTML
import base64
import os
import subprocess
import bisect
import av
import gc
from IPython.display import HTML
import base64


def display_video_clip(video_id, start_sec, end_sec, video_dir="/data/user_data/jamesdin/STAR/data/Charades_v1_480"):
    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    output_path = f"/tmp/{video_id}_{start_sec}_{end_sec}.mp4"

    # Extract clip using ffmpeg
    cmd = [
        "ffmpeg",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", video_path,
        "-c:v", "libx264",
        "-an",
        "-y",  # overwrite
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with open(output_path, "rb") as f:
        video_encoded = base64.b64encode(f.read()).decode("utf-8")

    html = f'''
    <video width="400" height="300" controls>
        <source src="data:video/mp4;base64,{video_encoded}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    '''
    return HTML(html)


def read_video_pyav(container, indices):
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

def read_video_pyav2(video_path, start, end, num_frames=8):
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


def visualize_video_attention(model, processor, prompt, video_frames, max_new_tokens=1):
    """
    Visualizes frame-level attention maps overlaid on video frames.

    Args:
        model: The transformer-based model.
        inputs: Tokenized inputs with input_ids and video tokens.
        video_frames: Numpy array of shape (num_frames, H, W, 3) or torch.Tensor.
    """

    inputs = processor(
        text=prompt, videos=video_frames, return_tensors="pt"
    ).to("cuda")
        
    # === Get attentions ===
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Only generate 1 token
            output_attentions=True,
            return_dict_in_generate=True
        )
        decoded = processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        print(decoded[0])

    attentions = outputs.attentions

    layer_idx = -1  # which layer attention output to look at
    token_idx = -1  # which generate token to look at
    last_layer_attn = attentions[token_idx][layer_idx][0]  # (num_heads, seq_len, seq_len)
    avg_attn = last_layer_attn.mean(dim=0)  # (seq_len, seq_len)
    last_token_attn = avg_attn[-1]  # (seq_len,)

    input_ids = inputs['input_ids'][0]
    video_token_id = 32001
    video_token_indices = (input_ids == video_token_id).nonzero(as_tuple=True)[0]
    video_token_start = video_token_indices[0].item()
    video_token_end = video_token_indices[-1].item() + 1
    video_attn_weights = last_token_attn[video_token_start:video_token_end]

    num_frames = video_frames.shape[0]
    num_tokens_per_frame = 257

    from torchvision.transforms import InterpolationMode
    
    clip_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(224),
    ])

    fig, axes = plt.subplots((num_frames + 3) // 4, 4, figsize=(16, 4 * ((num_frames + 3) // 4)))
    axes = axes.flatten()

    for frame_idx in range(num_frames):
        start_idx = frame_idx * num_tokens_per_frame + 1
        end_idx = start_idx + 256
        frame_token_attn = video_attn_weights[start_idx:end_idx]
        frame_attn_map = frame_token_attn.reshape(16, 16).cpu().numpy()
        heatmap = (frame_attn_map / (frame_attn_map.max() + 1e-8)).astype(np.float32)
        heatmap = cv2.resize(heatmap, (224, 224))

        # Get frame and apply clip transform
        frame_np = video_frames[frame_idx].cpu().numpy() if isinstance(video_frames, torch.Tensor) else video_frames[frame_idx]
        frame_pil = Image.fromarray((frame_np * 255).astype(np.uint8)) if frame_np.max() <= 1 else Image.fromarray(frame_np.astype(np.uint8))
        processed_frame = clip_preprocess(frame_pil)
        frame = processed_frame.numpy().transpose(1, 2, 0)
        frame = (frame * 255).astype(np.uint8)

        axes[frame_idx].imshow(frame)
        axes[frame_idx].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[frame_idx].axis('off')
        axes[frame_idx].set_title(f"Frame {frame_idx}")

    for j in range(num_frames, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

    del outputs
    torch.cuda.empty_cache()
    gc.collect()

def display_video_embed(video_id, video_dir="/data/user_data/jamesdin/STAR/data/Charades_v1_480"):
    video_path = os.path.join(video_dir, f"{video_id}.mp4")

    print("Embedding video:", video_path)
    
    with open(video_path, "rb") as f:
        video_encoded = base64.b64encode(f.read()).decode("utf-8")
    
    html = f'''
    <video width="320" height="240" controls>
        <source src="data:video/mp4;base64,{video_encoded}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    '''
    
    return HTML(html)

def display_sampled_frames(video_frames, frame_ids=None, title="Sampled Video Frames", frames_per_row=4):
    num_frames = len(video_frames)
    num_rows = (num_frames + frames_per_row - 1) // frames_per_row  # Ceiling division

    plt.figure(figsize=(frames_per_row * 4, num_rows * 4))  # Bigger figure size

    for i, frame in enumerate(video_frames):
        if isinstance(frame, torch.Tensor):
            frame = frame.permute(1, 2, 0).cpu().numpy()  # Convert (C, H, W) to (H, W, C)

        plt.subplot(num_rows, frames_per_row, i + 1)
        plt.imshow(frame.astype("uint8") if frame.max() > 1 else (frame * 255).astype("uint8"))
        if frame_ids is not None:
            plt.title(f"ID: {frame_ids[i]}")
        plt.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
