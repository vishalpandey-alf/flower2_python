# Blue Tulip 3D Botanical Animation (v2.4.0)

A Python-based 3D procedural botanical animation project that creates a cinematic, photorealistic Blue Tulip (*Tulipa*, stylized sapphire cultivar) blooming sequence on a light ivory studio backdrop with an interactive gesture-controlled web showcase.

## Project Structure

```text
flower2_python/
├── ffmpeg.exe
├── generate_tulip_animation.py
├── play.py
├── server.py
├── verify_metrics.py
├── tulip_landscape.mp4
├── tulip_portrait.mp4
├── templates/
│   └── index.html
├── prd.md
└── README.md
```

## About the Project

This project uses Python, NumPy, OpenCV, and FFmpeg to generate a stylized, photorealistic 3D botanical animation of a bouquet of 18 Blue Tulip heads (15 open layered blooms and 3 closed 6-tepal imbricate buds) across 14 caulescent stems, 18 coaxial peduncle necks, and 29 broad glaucous leaves. Each open flower features 9 two-sided sapphire tepals (3 outer, 3 inner, 3 ruffled petaloids) unfolded via pre-solved `P_states` poses, 6 golden-amber bilobed stamens, and a central 3-lobed pistil.

## Animation Stages

1. **Stem & Foliage Elongation (0.0s – 3.0s)** – 14 slender, curved green stems rise with 29 glaucous lanceolate leaves.
2. **Bud & Neck Formation (3.0s – 6.0s)** – Coaxial peduncle necks with receptacle domes connect stems to closed 6-tepal imbricate buds.
3. **Tepal Unfolding (6.0s – 9.0s)** – 9 tepals per open flower separate via continuous `P_states` interpolation with two-sided sapphire pigment and azure subsurface scattering.
4. **Androecium Maturation (9.0s – 13.0s)** – 6 golden-amber stamens mature and dehisce with pollen shimmer around the 3-lobed pistil.
5. **Full Radiant Anthesis & Orbital Reveal (13.0s – 18.0s)** – Full open bloom, orbital camera rotation, golden pollen motes, and drifting two-sided sapphire tepals.

## Technologies Used

- Python 3.10+ (NumPy, OpenCV)
- FFmpeg (bundled `ffmpeg.exe` or system PATH, all-intra `-g 1 -bf 0 -crf 18`)
- Three.js r183 (HDR EffectComposer, UnrealBloomPass, WebGL)
- MediaPipe Hands (Zero-latency camera hand-tracking)
- One Euro Filter & Spring Physics (Ultra-smooth gesture scrub)

## Requirements

- Python 3.10+
- NumPy
- OpenCV (`opencv-python`)
- Flask (`flask`)
- FFmpeg (bundled as `ffmpeg.exe` in project root)

Install the Python packages:

```bash
pip install -r requirements.txt
```

## How to Run

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
- **Living Post-Bloom Aura**: petals sway organically in a GLSL breeze alongside 3D floating sapphire tepals and animated background aurora gradients.
- Touch & pointer controls (2-finger pinch, 1-finger drag, wheel scrub, tap pause/resume) with `>= 44px` touch targets.

### 2. Desktop Viewer

```bash
python play.py
```

### 3. Render Full Animation

```bash
# Render 1920x1080 landscape:
python generate_tulip_animation.py --aspect landscape --out tulip_landscape.mp4

# Render 1080x1920 portrait:
python generate_tulip_animation.py --aspect portrait --out tulip_portrait.mp4
```

### 4. Quick Keyframe Previews & Automated Verification

```bash
# Export keyframe PNGs to ./previews/
python generate_tulip_animation.py --aspect portrait --preview

# Verify geometry connectivity, encode contract, and temporal stability
python verify_metrics.py --videos
```

## Video Specifications

- Resolution: 1920 × 1080 (Landscape) / 1080 × 1920 (Portrait)
- Frame rate: 30 FPS
- Duration: 18.0 seconds
- Total frames: 540 (All I-frames `-g 1 -bf 0`)
- Codec: H.264 / yuv420p / CRF 18 (`moov` before `mdat`)
- File size: ~44.5 MB (Landscape) / ~45.2 MB (Portrait)
