"""Interactive viewer for sakura_3d_botanical_animation.mp4."""
import os
import sys
from generate_sakura_animation import play_video

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(here, "sakura_3d_botanical_animation.mp4")
    play_video(video_path)

