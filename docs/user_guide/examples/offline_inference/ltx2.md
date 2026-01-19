# LTX-2 Text-To-Video

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/ltx2>.

The `Lightricks/LTX-2` pipeline generates high-quality videos from text prompts using a 19B parameter DiT architecture. It supports up to 4K video generation with synchronized audio.

## Local CLI Usage

```bash
python text_to_video.py \
  --prompt "A panda riding a bicycle through a forest, cinematic lighting" \
  --height 512 \
  --width 768 \
  --num_frames 121 \
  --num_inference_steps 40 \
  --guidance_scale 4.0 \
  --output ltx2_output.mp4
```

Key arguments:

- `--prompt`: Text description of the video to generate (string).
- `--height/--width`: Output resolution (defaults 512x768). Dimensions must be divisible by 32.
- `--num_frames`: Number of frames (should be 8*n+1, e.g., 25, 81, 121). Default is 121.
- `--guidance_scale`: Classifier-free guidance scale (default: 4.0). Range 3.0-5.0 recommended.
- `--negative_prompt`: Optional text describing what to avoid in the video.
- `--num_inference_steps`: Number of denoising steps (default: 40).
- `--seed`: Random seed for reproducibility (default: 42).
- `--fps`: Frames per second for the saved MP4 (default: 24).
- `--output`: Path to save the generated video.

## Example materials

??? abstract "text_to_video.py"
    ``````py
    --8<-- "examples/offline_inference/ltx2/text_to_video.py"
    ``````

??? abstract "text_to_video.md"
    ``````md
    --8<-- "examples/offline_inference/ltx2/text_to_video.md"
    ``````
