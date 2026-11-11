# Flower2 Python

A Python-based 3D botanical animation project that procedurally creates a cinematic Sakura (Japanese cherry blossom) blooming sequence.

## Project Structure

```text
flower2_python/
├── ffmpeg.exe
├── generate_sakura_animation.py
├── sakura_3d_botanical_animation.mp4
└── README.md
```

## About the Project

This project uses Python to generate a stylized, detailed 3D-inspired animation of a Sakura flower as it develops from a branch and buds into a full bloom.

## Animation Stages

1. **Branch Growth** – the branch begins to grow.
2. **Bud Formation** – flower buds appear.
3. **Petal Unfolding** – the petals gradually open.
4. **Full Bloom** – the flower reaches its completed form, with stamens and leaves.
5. **Cinematic Reveal** – drifting petals, bokeh, and camera movement complete the scene.

## Technologies Used

- Python
- NumPy
- OpenCV
- FFmpeg

## Files

- `generate_sakura_animation.py` – Python script used to generate the animation.
- `sakura_3d_botanical_animation.mp4` – rendered animation video.

## Requirements

- Python 3
- NumPy
- OpenCV (`opencv-python`)
- FFmpeg (bundled as `ffmpeg.exe` in project root, or via `imageio-ffmpeg` / system PATH)

Install the Python packages:

```bash
pip install numpy opencv-python imageio-ffmpeg
```

## How to Run

Open a terminal in the project folder:

```bash
cd flower2_python
```

### 1. Web Showcase (Localhost Website)

Start the interactive web server to view the animation in your browser:

```bash
python server.py
```

Then navigate to: **`http://localhost:5000`**

Features:
- High-res 9:16 video player with ambient sakura glow.
- Stage navigation bookmarks (jump directly to Stages 1–5).
- Speed controls (0.5x, 1x, 1.5x, 2x), scrubber, and keyframe preview gallery.

### 2. Desktop Video Player

To watch the rendered animation with an interactive looping player (Space to pause/resume, Q/Esc to quit):

```bash
python play.py
# or
python generate_sakura_animation.py --play
```

### 2. Render Full Animation

Run the animation script to render all 540 frames using the bundled FFmpeg:

```bash
python generate_sakura_animation.py
```

The rendered video is saved as:

```text
sakura_3d_botanical_animation.mp4
```

### 3. Quick Keyframe Previews

To render preview keyframes (PNGs at 2.2s, 5.2s, 8.8s, 12.6s, 16.2s):

```bash
python generate_sakura_animation.py --preview
```

## Video Specifications

- Resolution: 1080 × 1920
- Frame rate: 30 FPS
- Duration: 18 seconds
- Total frames: 540

## Author

Created by **Tanya**.
