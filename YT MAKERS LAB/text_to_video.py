from pathlib import Path
import random
import numpy as np
import torch
import streamlit as st
from diffusers import StableDiffusionPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"
_PIPELINE = None

STICKMAN_PROMPTS = [
    "A funny stickman tries to cook breakfast but ends up with a giant pancake on his head, bright cartoon style.",
    "A stickman slips on a banana peel, bounces around like a cartoon, and laughs at himself in a funny kitchen scene.",
    "A stickman comedian on stage tells jokes to a silly audience while juggling colorful balloons.",
    "A stickman and his pet dog get stuck in a tiny car and drive through a cartoon city in slow motion.",
    "A stickman tries to dance but keeps tripping over imaginary objects in a playful, upbeat scene.",
]


def get_stickman_prompt() -> str:
    return random.choice(STICKMAN_PROMPTS)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_text_to_video_pipeline(device: str = None) -> StableDiffusionPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    device = device or get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    _PIPELINE = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )
    if device == "cpu":
        _PIPELINE.enable_attention_slicing()
    _PIPELINE = _PIPELINE.to(device)
    return _PIPELINE


def generate_video(prompt: str, output_path: str = "yt_makers_lab_output.mp4", num_inference_steps: int = 30, guidance_scale: float = 7.5, num_frames: int = 4, fps: int = 2) -> Path:
    """Generate a short video from text by creating multiple image frames and saving as MP4."""
    pipe = load_text_to_video_pipeline()
    frames = []

    for frame_index in range(num_frames):
        frame_prompt = f"{prompt}, cinematic frame {frame_index + 1}"
        result = pipe(frame_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        image = result.images[0]
        frames.append(np.array(image))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    import imageio
    imageio.mimwrite(str(output_file), frames, fps=fps)
    
    video_duration = num_frames / fps
    return output_file

    return output_file


def run_streamlit_ui():
    st.set_page_config(page_title="YT MAKERS LAB Text-to-Video", layout="wide")
    st.title("YT MAKERS LAB Text-to-Video")
    st.write("Generate a short video-like MP4 by creating multiple Stable Diffusion XL frames.")

    device = get_device()
    st.info(f"Device: {device}")
    if device == "cpu":
        st.warning("No CUDA GPU detected. Generation will be very slow on CPU.")

    prompt = st.text_area("Prompt", value="A funny stickman comedy scene", height=160)
    if st.button("Use Stickman Comedy Prompt"):
        prompt = get_stickman_prompt()

    output_name = st.text_input("Output file", value="yt_makers_lab_output.mp4")
    steps = st.slider("Inference steps", 10, 50, 30, 1)
    guidance = st.slider("Guidance scale", 1.0, 15.0, 7.5, 0.5)
    num_frames = st.slider("Number of frames", 2, 50, 12, 1)
    fps = st.slider("Frames per second (FPS)", 1, 10, 2, 1)
    
    st.info(f"Estimated video duration: {num_frames / fps:.1f} seconds")

    if st.button("Generate Video"):
        if not prompt.strip():
            st.error("Please enter a prompt before generating.")
        else:
            with st.spinner("Generating frames and building video..."):
                try:
                    output_file = generate_video(
                        prompt=prompt,
                        output_path=output_name,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        num_frames=num_frames,
                        fps=fps,
                    )
                    st.success(f"Saved video to: {output_file}")
                    st.video(str(output_file))
                except Exception as exc:
                    st.error(f"Error: {exc}")


if __name__ == "__main__":
    run_streamlit_ui()

