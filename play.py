"""Interactive desktop viewer for Tulip Botanical Animation."""
import os
import sys

def play():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "tulip_landscape.mp4",
        "tulip_portrait.mp4",
    ]
    target = None
    for c in candidates:
        full = os.path.join(here, c)
        if os.path.isfile(full):
            target = full
            break
    if not target:
        print("No animation video file found to play.")
        sys.exit(1)

    import cv2
    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        print(f"Error opening video: {target}")
        sys.exit(1)

    window_name = "Tulip 3D Botanical Animation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = int(1000.0 / fps)
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cv2.imshow(window_name, frame)

        key = cv2.waitKey(1 if paused else delay) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key == 32:  # Space
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play()
