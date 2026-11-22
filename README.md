# Blue Tulip 3D Botanical Animation

A Python-based 3D procedural botanical animation project that creates a cinematic, photorealistic Blue Tulip (*Tulipa*, stylized sapphire cultivar) blooming sequence with an interactive gesture-controlled web showcase.

## Project Structure

```text
flower2_python/
├── ffmpeg.exe
├── generate_tulip_animation.py
├── tulip_landscape.mp4
├── tulip_portrait.mp4
├── server.py
├── templates/
│   └── index.html
├── prd.md
└── README.md
```

## About the Project

This project uses Python, NumPy, OpenCV, and FFmpeg to generate a stylized, photorealistic 3D botanical animation of a cluster of 7 Blue Tulips as they grow from slender, curved stems into full, radiant 6-tepal anthesis with golden-amber stamens and central pistils.

## Animation Stages

1. **Stem & Foliage Elongation (0.0s – 3.0s)** – 7 slender, curved green stems rise with 14 glaucous lanceolate leaves.
2. **Bud Formation (3.0s – 6.0s)** – Closed teardrop buds form with cyan/pale-green apical tints.
3. **Tepal Unfolding (6.0s – 9.0s)** – 6 tepals (3 outer, 3 inner) separate with ogee contour, translucent subsurface scattering, and subtle venation.
4. **Androecium Maturation (9.0s – 13.0s)** – 6 golden-amber stamens (#E8B923) mature and dehisce with pollen around the 3-lobed pistil.
5. **Full Radiant Anthesis & Orbital Reveal (13.0s – 18.0s)** – Full open bloom, orbital camera rotation, floating pollen and drifting sapphire tepals.

## Technologies Used

- Python 3 (NumPy, OpenCV)
- FFmpeg (bundled `ffmpeg.exe` or system PATH, all-intra `-g 1 -bf 0 -crf 18`)
- Three.js r183 (HDR EffectComposer, UnrealBloomPass, WebGL)
- MediaPipe Hands (Zero-latency camera hand-tracking)
- One Euro Filter & Spring Physics (Ultra-smooth gesture scrub)

## Requirements

- Python 3
- NumPy
- OpenCV (`opencv-python`)
- Flask (`flask`)
- FFmpeg (bundled as `ffmpeg.exe` in project root)

Install the Python packages:

```bash
pip install numpy opencv-python flask imageio-ffmpeg
```

## How to Run

Open a terminal in the project folder:

### 1. Web Showcase (Localhost Website)

Start the interactive web server:

```bash
python server.py
```

Then navigate to: **`http://localhost:5000`**

Features:
- Responsive full-screen viewport (`tulip_landscape.mp4` on PC/landscape, `tulip_portrait.mp4` on mobile/portrait) with all-intra frame encoding (`-g 1 -bf 0`) for zero-lag scrubbing.
- **3D HDR WebGL Compositor (Three.js `EffectComposer` + `UnrealBloomPass` + GSAP)**: real-time 3D double-helix sapphire/gold stardust spiral & luminous orbital ribbons that steer, tilt, and flare with hand or pointer gestures.
- **One Euro Filter + Spring Physics Hand Tracking**: adaptive smoothing over MediaPipe Hands pinch & spread for zero-jitter, silky blossom opening.
- **Living Post-Bloom Aura**: once bloomed via hand gesture, petals sway organically in a GLSL breeze alongside 3D floating sapphire tepals, radiant edge glow emission, and animated background sparkle aurora gradients (midnight navy, teal, sapphire, violet) until the next hand gesture.
- Touch & pointer controls (2-finger pinch, 1-finger drag, wheel scrub, tap pause/resume) with `>= 44px` touch targets.

### 2. Render Full Animation

To render the 540-frame landscape or portrait video:

```bash
# Render 1920x1080 landscape:
python generate_tulip_animation.py --aspect landscape --out tulip_landscape.mp4

# Render 1080x1920 portrait:
python generate_tulip_animation.py --aspect portrait --out tulip_portrait.mp4
```

### 3. Quick Keyframe Previews

To render preview keyframes (PNGs at 0s, 3s, 6s, 9s, 13s, 18s):

```bash
python generate_tulip_animation.py --preview
```

## Video Specifications

- Resolution: 1920 × 1080 (Landscape) / 1080 × 1920 (Portrait)
- Frame rate: 30 FPS
- Duration: 18.0 seconds
- Total frames: 540 (All I-frames `-g 1 -bf 0`)
- Codec: H.264 / yuv420p / CRF 18
- Bitrate: Variable, all-intra, < 100 MB
