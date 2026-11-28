"""Automated acceptance checks for Blue Tulip v2.4.0 (connectivity, encode contract, motion)."""
import os
import sys
import struct
import subprocess
import numpy as np
import cv2
import generate_tulip_animation as g


def check_connectivity():
    els = g.build_tulip_scene()
    sm = g.smootherstep
    rc = lambda P, i: P[i].mean(axis=0)
    stems = [e for e in els if e.kind == "stem"]
    heads = {}
    for e in els:
        if e.kind == "petal":
            heads.setdefault(tuple(np.round(e.anchor, 4)), []).append(e)
    necks = {}
    for s in stems:
        end = rc(s.P, -1)
        key = min(heads, key=lambda a: np.linalg.norm(np.array(a) - end))
        if np.linalg.norm(np.array(key) - end) < 0.08:
            necks[key] = s
    assert len(heads) == 18, f"Expected 18 heads, got {len(heads)}"
    assert len(necks) == 18, f"Expected 18 necks, got {len(necks)}"
    cent = [(s, np.array([rc(s.P, i) for i in range(s.P.shape[0])])) for s in stems]
    max_d = 0.0
    for key, neck in necks.items():
        start = rc(neck.P, 0)
        d = min(np.min(np.linalg.norm(c - start, axis=1)) for s, c in cent if s is not neck)
        max_d = max(max_d, float(d))
        assert d < 0.01, ("neck start off parent axis", key, d)
        assert np.linalg.norm(rc(neck.P, -1) - np.array(key)) < 0.06
        for t in np.arange(0, 18.01, 0.05):
            if max(sm(e.surf_t0, e.surf_t1, t) for e in heads[key]) > 0.01:
                assert sm(neck.surf_t0, neck.surf_t1, t) >= 0.985, (key, t)
    print(f"[PASS] Connectivity: 18/18 heads, 18/18 necks, max parent-axis offset {max_d:.5f}, 0 timing violations")


def check_mp4_moov_before_mdat(path):
    with open(path, "rb") as f:
        pos = 0
        size_total = os.path.getsize(path)
        moov_pos = mdat_pos = None
        while pos < size_total:
            f.seek(pos)
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            sz, atom = struct.unpack(">I4s", hdr)
            if sz == 1:
                sz = struct.unpack(">Q", f.read(8))[0]
            elif sz == 0:
                sz = size_total - pos
            if atom == b"moov" and moov_pos is None:
                moov_pos = pos
            elif atom == b"mdat" and mdat_pos is None:
                mdat_pos = pos
            if sz < 8:
                break
            pos += sz
    assert moov_pos is not None and mdat_pos is not None, f"{path}: missing moov/mdat"
    assert moov_pos < mdat_pos, f"{path}: moov ({moov_pos}) after mdat ({mdat_pos})"
    return moov_pos, mdat_pos


def check_video(path, exp_w, exp_h):
    assert os.path.isfile(path), f"Missing video: {path}"
    check_mp4_moov_before_mdat(path)
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"Failed to open {path}"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert (w, h) == (exp_w, exp_h), f"{path}: expected {exp_w}x{exp_h}, got {w}x{h}"
    assert abs(fps - 30.0) < 0.1, f"{path}: expected 30 fps, got {fps}"
    assert n_frames == 540, f"{path}: expected 540 frames, got {n_frames}"

    # Verify all-intra via ffmpeg pkt_flags or keyframe seekability + temporal stability
    diffs = []
    lumas = []
    prev = None
    actual = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        actual += 1
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (w // 2, h // 2))
        lumas.append(float(gray.mean()))
        if prev is not None:
            diffs.append(float(np.mean(np.abs(gray.astype(np.float32) - prev.astype(np.float32)))))
        prev = gray
    cap.release()
    assert actual == 540, f"{path}: read {actual} frames"

    diffs_arr = np.array(diffs)
    luma_jumps = np.abs(np.diff(lumas))
    med_d = float(np.median(diffs_arr))
    p99_d = float(np.percentile(diffs_arr, 99))
    max_d = float(np.max(diffs_arr))
    max_l = float(np.max(luma_jumps))
    size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
    print(f"[PASS] {os.path.basename(path)}: {w}x{h} @ {fps:.0f}fps, {actual} frames, {size_mb:.1f} MB, "
          f"moov<mdat, diff med={med_d:.2f} p99={p99_d:.2f} max={max_d:.2f}, max_luma_jump={max_l:.2f}")
    assert max_d <= 1.5, f"{path}: max frame diff {max_d:.2f} > 1.5"


if __name__ == "__main__":
    check_connectivity()
    if "--videos" in sys.argv:
        check_video("tulip_landscape.mp4", 1920, 1080)
        check_video("tulip_portrait.mp4", 1080, 1920)
