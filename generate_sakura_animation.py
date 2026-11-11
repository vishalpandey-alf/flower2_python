#!/usr/bin/env python3
"""
Cinematic 3D Cherry Blossom (Sakura / Prunus serrulata) Botanical Animation
===========================================================================
Generates a realistic 9:16 (1080x1920 @ 30fps, 18.0s) 3D botanical animation
of a flowering Japanese Cherry Blossom branch growing and blooming step by step:
  - Stage 1 (0.0s - 3.6s): Bare gnarled woody branch & secondary twigs growing
                           with procedural mahogany bark, knots, and lenticels.
  - Stage 2 (3.0s - 6.6s): Corymb pedicels, crimson-green calyx tubes (hypanthia),
                           and swelling rose-pink flower buds at multiple stages.
  - Stage 3 (6.2s - 10.8s): Individual 5-petal construction & unfolding per blossom
                           with authentic bilobed apical notch (emarginate cleft),
                           concave cupping, radial venation, and SSS shading.
  - Stage 4 (10.0s - 14.2s): Full bloom across depths, 30 radial stamens with
                           delicate golden-amber bilobed anthers, and serrated
                           young bronze-green Sakura leaves unfolding.
  - Stage 5 (13.6s - 18.0s): Cinematic final reveal with 3D drifting Sakura petals,
                           depth-of-field bokeh, and smooth 3D camera parallax.
"""

import os
import sys
import math
import time
import shutil
import subprocess
import multiprocessing as mp
import numpy as np
import cv2

# ============================================================================
# VIDEO & RENDER CONFIGURATION (9:16 Vertical Reel)
# ============================================================================
WIDTH = 1080
HEIGHT = 1920
FPS = 30
DURATION = 18.0
TOTAL_FRAMES = int(FPS * DURATION)  # 540 frames
SHIFT = 3
SUBPIX = 1 << SHIFT  # 8x subpixel anti-aliased rasterization
SHOW_HUD = False     # Clean render without telemetry text overlay by default


# ============================================================================
# MATHEMATICAL, SPLINE & EASING UTILITIES
# ============================================================================
def clamp(x, lo=0.0, hi=1.0):
    return np.clip(x, lo, hi)


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / np.maximum(1e-8, (edge1 - edge0)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0, edge1, x):
    t = np.clip((x - edge0) / np.maximum(1e-8, (edge1 - edge0)), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def normalize_v3(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-8)


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def build_basis_from_dir(direction, up_hint=np.array([0.0, 1.0, 0.0], dtype=np.float32)):
    """Construct an orthonormal 3x3 basis matrix whose local Y-axis aligns with `direction`."""
    y_axis = direction / max(1e-8, np.linalg.norm(direction))
    if abs(np.dot(y_axis, up_hint)) > 0.95:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_axis = np.cross(up_hint, y_axis)
    x_axis /= max(1e-8, np.linalg.norm(x_axis))
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(1e-8, np.linalg.norm(z_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=-1).astype(np.float32)


def eval_cubic_bezier(p0, p1, p2, p3, u):
    """Evaluate 3D cubic Bezier spline and tangent vector for parameter array u in [0, 1]."""
    u = np.asarray(u, dtype=np.float32)[..., None]
    om = 1.0 - u
    pos = (om ** 3) * p0 + 3.0 * (om ** 2) * u * p1 + 3.0 * om * (u ** 2) * p2 + (u ** 3) * p3
    tan = 3.0 * (om ** 2) * (p1 - p0) + 6.0 * om * u * (p2 - p1) + 3.0 * (u ** 2) * (p3 - p2)
    return pos, tan


def compute_bishop_frame(T):
    """Compute a twist-free parallel transport frame (N1, N2) along tangent sequence T (Nu, 3)."""
    Nu = len(T)
    N1 = np.zeros_like(T)
    N2 = np.zeros_like(T)
    t0 = T[0]
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(t0, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    n1 = np.cross(t0, ref)
    n1 /= max(1e-8, np.linalg.norm(n1))
    n2 = np.cross(t0, n1)
    N1[0], N2[0] = n1, n2

    for i in range(1, Nu):
        ti = T[i]
        n1_proj = n1 - np.dot(n1, ti) * ti
        norm_n1 = np.linalg.norm(n1_proj)
        if norm_n1 < 1e-6:
            n1_proj = np.cross(ti, n2)
            norm_n1 = np.linalg.norm(n1_proj)
        n1 = n1_proj / max(1e-8, norm_n1)
        n2 = np.cross(ti, n1)
        N1[i], N2[i] = n1, n2
    return N1, N2


# ============================================================================
# 3D BOTANICAL SURFACE ELEMENT
# ============================================================================
class BotanicalElement:
    """
    Represents a 3D parametric botanical mesh P(u, v) of shape (Nu, Nv, 3)
    with continuous longitudinal growth (along u), optional 3D morphological
    unfolding (from P_closed to P_open), color maturation (C_closed to C_open),
    scale growth from an anchor point, and physically-based SSS + specular shading.
    """
    def __init__(self, kind, P, C, wire_t0, wire_t1, surf_t0, surf_t1,
                 sss_strength=0.35, spec_strength=0.22, shininess=18.0,
                 wire_color=(0.95, 0.78, 0.62), wire_stride_u=4, wire_stride_v=3,
                 P_closed=None, C_closed=None, bloom_t0=None, bloom_t1=None,
                 extra_wire_t0=None, extra_wire_t1=None,
                 anchor=None, scale_t0=None, scale_t1=None, scale_init=1.0,
                 dof_layer="mid", pollen_tip=None):
        self.kind = kind
        self.P = P.astype(np.float32)
        self.P_closed = P_closed.astype(np.float32) if P_closed is not None else None
        self.C = np.clip(C, 0.0, 1.0).astype(np.float32)
        self.C_closed = np.clip(C_closed, 0.0, 1.0).astype(np.float32) if C_closed is not None else None
        self.wire_t0 = wire_t0
        self.wire_t1 = wire_t1
        self.surf_t0 = surf_t0
        self.surf_t1 = surf_t1
        self.sss_strength = sss_strength
        self.spec_strength = spec_strength
        self.shininess = shininess
        self.wire_color = np.array(wire_color, dtype=np.float32)
        self.wire_stride_u = wire_stride_u
        self.wire_stride_v = wire_stride_v
        self.bloom_t0 = bloom_t0 if bloom_t0 is not None else surf_t0
        self.bloom_t1 = bloom_t1 if bloom_t1 is not None else surf_t1
        self.extra_wire_t0 = extra_wire_t0
        self.extra_wire_t1 = extra_wire_t1
        self.anchor = np.asarray(anchor, dtype=np.float32) if anchor is not None else None
        self.scale_t0 = scale_t0
        self.scale_t1 = scale_t1
        self.scale_init = scale_init
        self.dof_layer = dof_layer  # "fg", "mid", "bg"
        self.pollen_tip = pollen_tip  # Optional (pos_closed, pos_open, rad_px, bgr_core, bgr_rim)

    def evaluate_mesh(self, t, wind_vec=None):
        """Evaluate current 3D positions P(u, v) and colors C(u, v) at animation time t."""
        if self.P_closed is not None:
            b_prog = smootherstep(self.bloom_t0, self.bloom_t1, t)
            P = self.P_closed * (1.0 - b_prog) + self.P * b_prog
        else:
            b_prog = 1.0
            P = self.P.copy()

        if self.C_closed is not None:
            C = self.C_closed * (1.0 - b_prog) + self.C * b_prog
        else:
            C = self.C

        # Optional 3D scale swelling from anchor point (e.g. buds & leaves growing)
        if self.anchor is not None and self.scale_t0 is not None:
            sc_prog = smootherstep(self.scale_t0, self.scale_t1, t)
            sc = self.scale_init + (1.0 - self.scale_init) * sc_prog
            if sc < 0.999:
                P = self.anchor + (P - self.anchor) * sc

        # Subtle organic breeze sway on petals, leaves, and stamens
        if wind_vec is not None and self.anchor is not None and self.kind in ("petal", "leaf", "stamen"):
            dist_from_anchor = np.linalg.norm(P - self.anchor, axis=-1, keepdims=True)
            sway_factor = (dist_from_anchor ** 1.35) * (0.28 if self.kind == "petal" else 0.22)
            P = P + sway_factor * wind_vec

        return P, C, b_prog


def slice_mesh_along_u(P, C, prog):
    """
    Smoothly slice a parametric mesh (Nu, Nv, 3) along the longitudinal u-axis
    for `prog` in (0, 1], interpolating the exact fractional wavefront tip row
    so growth is completely continuous with zero stair-stepping.
    """
    Nu = P.shape[0]
    if prog >= 0.999:
        p00, p10, p11, p01 = P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]
        d_u = 0.5 * ((p10 - p00) + (p11 - p01))
        d_v = 0.5 * ((p01 - p00) + (p11 - p10))
        normals = normalize_v3(np.cross(d_v, d_u))
        c_quad = 0.25 * (C[:-1, :-1] + C[1:, :-1] + C[1:, 1:] + C[:-1, 1:])
        u_quad = np.linspace(0.0, 1.0, Nu - 1, dtype=np.float32)[:, None]
        return P, normals, c_quad, u_quad

    exact_u = max(1.02, prog * (Nu - 1))
    k = int(math.floor(exact_u))
    frac = exact_u - k
    if k >= Nu - 1:
        k = Nu - 2
        frac = 1.0

    P_sub = P[:k + 2].copy()
    C_sub = C[:k + 2].copy()
    if frac < 0.999:
        P_sub[k + 1] = P[k] * (1.0 - frac) + P[k + 1] * frac
        C_sub[k + 1] = C[k] * (1.0 - frac) + C[k + 1] * frac

    p00, p10, p11, p01 = P_sub[:-1, :-1], P_sub[1:, :-1], P_sub[1:, 1:], P_sub[:-1, 1:]
    d_u = 0.5 * ((p10 - p00) + (p11 - p01))
    d_v = 0.5 * ((p01 - p00) + (p11 - p10))
    normals = normalize_v3(np.cross(d_v, d_u))
    c_quad = 0.25 * (C_sub[:-1, :-1] + C_sub[1:, :-1] + C_sub[1:, 1:] + C_sub[:-1, 1:])
    u_quad = (np.arange(k + 1, dtype=np.float32) / max(1, Nu - 1))[:, None]
    return P_sub, normals, c_quad, u_quad


# ============================================================================
# PROCEDURAL 3D BOTANICAL GENERATORS (PRUNUS SERRULATA / SAKURA)
# ============================================================================
def generate_woody_branch(p0, p1, p2, p3, r_base, r_tip, Nu, Nv,
                          wire_t0, wire_t1, surf_t0, surf_t1,
                          knot_u_list=None, rng_seed=42, dof_layer="mid"):
    """
    Generate a high-resolution 3D woody Sakura branch along a cubic Bezier spline:
      - Organic tapering, natural nodal swellings (knots) at branch junctions
      - Longitudinal bark ridges and micro-furrows
      - Smooth procedural Prunus mahogany-brown bark with horizontal lenticels
    """
    rng = np.random.RandomState(rng_seed)
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')
    theta = (V + 1.0) * math.pi

    pos, tan = eval_cubic_bezier(p0, p1, p2, p3, u)
    T = normalize_v3(tan)
    N1, N2 = compute_bishop_frame(T)

    # Bark color palette (RGB in [0, 1]): deep reddish-mahogany to warm charcoal-brown
    c_bark_deep = np.array([0.17, 0.10, 0.09], dtype=np.float32)
    c_bark_mid = np.array([0.31, 0.18, 0.15], dtype=np.float32)
    c_bark_warm = np.array([0.43, 0.25, 0.20], dtype=np.float32)
    c_lenticel = np.array([0.62, 0.47, 0.39], dtype=np.float32)
    c_lichen = np.array([0.34, 0.37, 0.26], dtype=np.float32)

    P = np.zeros((Nu, Nv, 3), dtype=np.float32)
    C = np.zeros((Nu, Nv, 3), dtype=np.float32)

    phase1 = rng.uniform(0, 2 * math.pi)
    phase2 = rng.uniform(0, 2 * math.pi)

    for i in range(Nu):
        ui = u[i]
        r = r_base * ((1.0 - ui) ** 0.86) + r_tip * (ui ** 0.86)

        if knot_u_list:
            for ku, k_amp in knot_u_list:
                r *= 1.0 + k_amp * math.exp(-((ui - ku) / 0.06) ** 2)

        bark_rough = (
            1.0
            + 0.042 * np.cos(5.0 * theta[i] + 3.2 * ui + phase1)
            + 0.025 * np.sin(9.0 * theta[i] - 5.5 * ui + phase2)
            + 0.015 * np.cos(14.0 * theta[i] + 10.0 * ui)
        )
        rad = r * bark_rough

        gnarl_1 = 0.014 * r_base * math.sin(ui * 4.0 * math.pi + phase1)
        gnarl_2 = 0.014 * r_base * math.cos(ui * 5.0 * math.pi + phase2)
        center_i = pos[i] + gnarl_1 * N1[i] + gnarl_2 * N2[i]

        P[i] = center_i + rad[:, None] * (
            np.cos(theta[i])[:, None] * N1[i] + np.sin(theta[i])[:, None] * N2[i]
        )

        ridge_val = 0.5 + 0.5 * np.cos(5.0 * theta[i] + 3.2 * ui + phase1)
        fine_val = 0.5 + 0.5 * np.sin(11.0 * theta[i] - 7.5 * ui + phase2)
        base_col = (
            (1.0 - ridge_val)[:, None] * c_bark_deep
            + ridge_val[:, None] * c_bark_mid
            + 0.24 * fine_val[:, None] * (c_bark_warm - c_bark_mid)
        )
        tip_warmth = smoothstep(0.65, 1.0, ui) * 0.26
        base_col = (1.0 - tip_warmth) * base_col + tip_warmth * c_bark_warm
        C[i] = base_col

    # Smooth Gaussian-profile horizontal lenticels (characteristic of cherry bark)
    n_lenticels = int(Nu * 0.75)
    for _ in range(n_lenticels):
        u_c = rng.uniform(0.05, 0.92)
        v_c = rng.uniform(-0.85, 0.85)
        u_w = rng.uniform(0.008, 0.016)
        v_w = rng.uniform(0.08, 0.18)
        strength = rng.uniform(0.35, 0.65)
        mask = strength * np.exp(-(((U - u_c) / u_w) ** 2 + ((V - v_c) / v_w) ** 2))[..., None]
        C = (1.0 - mask) * C + mask * c_lenticel

    if r_base > 0.06:
        lichen_field = np.clip(np.sin(3.0 * theta + 4.5 * U + phase1) - 0.55, 0.0, 1.0)[..., None]
        u_fade = (1.0 - smoothstep(0.35, 0.75, U))[..., None]
        C = (1.0 - 0.35 * lichen_field * u_fade) * C + (0.35 * lichen_field * u_fade) * c_lichen

    el = BotanicalElement(
        kind="branch", P=P, C=C,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.06, spec_strength=0.16, shininess=12.0,
        wire_color=(0.92, 0.68, 0.45), wire_stride_u=6, wire_stride_v=4,
        dof_layer=dof_layer
    )
    return el, pos, T, N1, N2


def generate_pedicel_and_calyx(p0, p1, p2, p3, scale, wire_t0, wire_t1, surf_t0, surf_t1, dof_layer="mid"):
    """
    Generate a slender cherry blossom pedicel (flower stalk) that flares into
    the characteristic tubular wine-red/green hypanthium (calyx cup) at its tip.
    """
    Nu, Nv = 24, 15
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    theta = (v + 1.0) * math.pi

    pos, tan = eval_cubic_bezier(p0, p1, p2, p3, u)
    T = normalize_v3(tan)
    N1, N2 = compute_bishop_frame(T)

    c_stalk_base = np.array([0.32, 0.45, 0.20], dtype=np.float32)
    c_stalk_mid = np.array([0.42, 0.34, 0.22], dtype=np.float32)
    c_calyx_cup = np.array([0.46, 0.15, 0.21], dtype=np.float32)

    P = np.zeros((Nu, Nv, 3), dtype=np.float32)
    C = np.zeros((Nu, Nv, 3), dtype=np.float32)

    for i in range(Nu):
        ui = u[i]
        if ui < 0.72:
            r = scale * (0.015 * (1.0 - ui / 0.72) + 0.013 * (ui / 0.72))
        else:
            cu = (ui - 0.72) / 0.28
            r = scale * (0.013 + 0.036 * math.sin(cu * 0.82 * math.pi))

        rib = 1.0 + 0.04 * np.cos(5.0 * theta) * smoothstep(0.65, 1.0, ui)
        rad = r * rib
        P[i] = pos[i] + rad[:, None] * (np.cos(theta)[:, None] * N1[i] + np.sin(theta)[:, None] * N2[i])

        if ui < 0.65:
            mix = ui / 0.65
            col = (1.0 - mix) * c_stalk_base + mix * c_stalk_mid
        else:
            mix = (ui - 0.65) / 0.35
            col = (1.0 - mix) * c_stalk_mid + mix * c_calyx_cup
        striation = 0.94 + 0.06 * np.cos(5.0 * theta)[:, None]
        C[i] = col * striation

    el = BotanicalElement(
        kind="pedicel", P=P, C=C,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.18, spec_strength=0.20, shininess=16.0,
        wire_color=(0.55, 0.85, 0.48), wire_stride_u=5, wire_stride_v=4,
        dof_layer=dof_layer
    )
    return el, pos[-1], T[-1]


def generate_sakura_bud(tip_pos, tip_dir, scale, stage, wire_t0, wire_t1, surf_t0, surf_t1, rng_seed=10, dof_layer="mid"):
    """
    Generate a 3D cherry blossom bud at different developmental stages:
      - stage="tight": Small crimson-green closed winter/early-spring bud
      - stage="swelling": Plump ovoid rose-pink bud with visible green-crimson calyx sepals
      - stage="cracking": Full pink bud with slightly parting spiral petal tips
    """
    basis = build_basis_from_dir(tip_dir)
    Nu, Nv = 36, 23
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')
    theta = (V + 1.0) * math.pi

    if stage == "tight":
        length = 0.145 * scale
        r_max = 0.050 * scale
    elif stage == "swelling":
        length = 0.215 * scale
        r_max = 0.078 * scale
    else:  # "cracking"
        length = 0.255 * scale
        r_max = 0.094 * scale

    env = np.maximum(0.0, np.sin(math.pi * (U ** 0.62))) ** 0.82
    imbricate = 1.0 + 0.065 * np.cos(5.0 * theta + 1.8 * U) * smoothstep(0.15, 0.90, U)
    r_prof = r_max * env * imbricate

    if stage == "cracking":
        flare_tip = 0.015 * scale * (smoothstep(0.78, 1.0, U) ** 1.5) * (0.6 + 0.4 * np.cos(5.0 * theta))
        r_prof = r_prof + flare_tip

    y_loc = length * U
    x_loc = r_prof * np.cos(theta)
    z_loc = r_prof * np.sin(theta)
    P_local = np.stack([x_loc, y_loc, z_loc], axis=-1)
    P_world = tip_pos + P_local @ basis.T

    c_calyx_green = np.array([0.30, 0.42, 0.18], dtype=np.float32)
    c_calyx_wine = np.array([0.48, 0.15, 0.22], dtype=np.float32)
    c_bud_deep = np.array([0.82, 0.30, 0.46], dtype=np.float32)
    c_bud_soft = np.array([0.93, 0.58, 0.70], dtype=np.float32)
    c_bud_tip = np.array([0.97, 0.76, 0.84], dtype=np.float32)

    sepal_reach = 0.52 if stage == "tight" else (0.38 if stage == "swelling" else 0.28)
    sepal_lobes = sepal_reach + 0.12 * np.cos(5.0 * theta)
    sepal_mask = (1.0 - smoothstep(sepal_lobes - 0.08, sepal_lobes + 0.06, U))[..., None]

    calyx_col = (1.0 - smoothstep(0.0, sepal_reach, U)[..., None]) * c_calyx_green + smoothstep(0.0, sepal_reach, U)[..., None] * c_calyx_wine
    petal_u = np.clip((U - sepal_reach * 0.7) / max(1e-4, 1.0 - sepal_reach * 0.7), 0.0, 1.0)[..., None]
    petal_col = (1.0 - petal_u) * c_bud_deep + petal_u * c_bud_soft
    petal_col = (1.0 - (petal_u ** 2)) * petal_col + (petal_u ** 2) * c_bud_tip

    vein = (0.94 + 0.06 * np.cos(15.0 * theta + 2.0 * U))[..., None]
    C = (sepal_mask * calyx_col + (1.0 - sepal_mask) * petal_col) * vein

    return BotanicalElement(
        kind="bud", P=P_world, C=C,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.32, spec_strength=0.22, shininess=22.0,
        wire_color=(0.95, 0.58, 0.72), wire_stride_u=6, wire_stride_v=4,
        anchor=tip_pos, scale_t0=surf_t0, scale_t1=surf_t1 + 0.8, scale_init=0.25,
        dof_layer=dof_layer
    )


def generate_sakura_petal(center, basis, azimuth, scale, max_openness, color_tone,
                          wire_t0, wire_t1, surf_t0, surf_t1, bloom_t0, bloom_t1,
                          rng_seed=100, dof_layer="mid"):
    """
    Generate a single realistic 3D Cherry Blossom (Sakura) petal with:
      - Authentic obovate fan geometry where u in [0, 1] sweeps from the narrow
        basal claw out to the bilobed distal margin
      - Distinct V-shaped apical cleft / notch (emarginate tip at v = 0, u -> 1)
        flanked by two plush, rounded apical lobes (|v| ~ 0.38)
      - Proper angular span so all 5 petals per flower are individually distinct
      - Full 3D morphological unfolding: forms a plump deep-pink closed bud in Stage 2
        (P_closed, C_closed) and unfolds petal-by-petal into a cupped blush blossom in Stage 3
    """
    rng = np.random.RandomState(rng_seed)
    Nu, Nv = 46, 33
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')
    abs_V = np.abs(V)

    # Subtle per-petal organic variation
    len_var = rng.uniform(0.95, 1.05)
    wid_var = rng.uniform(0.94, 1.06)
    az_jitter = rng.uniform(-0.04, 0.04)
    roll_twist = rng.uniform(-0.07, 0.07)
    wave_phase = rng.uniform(0.0, 2.0 * math.pi)

    length = 0.37 * scale * len_var
    # Tuned half-width so all 5 rounded obovate petals are clearly distinct with natural gaps at the tips
    w_max = 0.148 * scale * wid_var

    # Elliptical perimeter parameterization across v in [-1, 1]:
    # theta_p sweeps from -0.84*pi (left claw base) through 0 (apical notch) to +0.84*pi (right claw base)
    theta_p = V * (0.84 * math.pi)
    abs_V = np.abs(V)

    # Smooth oval longitudinal contour + twin rounded apical lobes + central V-cleft notch
    s_oval = 0.50 + 0.50 * np.cos(theta_p)
    lobe_boost = 0.065 * np.exp(-(((abs_V - 0.23) / 0.15) ** 2))
    cleft_notch = 0.175 * (np.maximum(0.0, 1.0 - (abs_V / 0.17)) ** 1.45)
    S_edge = length * np.maximum(0.02, s_oval + lobe_boost - cleft_notch)

    # Smooth obovate width contour (widest in the upper-middle belly, rounded over the lobes)
    X_edge = w_max * np.sin(theta_p) * (1.0 + 0.14 * np.cos(theta_p))

    # Interior blade coordinates swept smoothly from the narrow basal claw (u=0) to the perimeter (u=1)
    u_r = U ** 0.88
    s_blade = (1.0 - u_r) * (0.015 * length * (1.0 - V ** 2)) + u_r * S_edge
    x_blade = (1.0 - u_r) * (0.09 * w_max * V) + u_r * X_edge

    def build_petal_state(open_amt):
        # Normalized arc position along petal
        s_norm = np.clip(s_blade / max(1e-5, length), 0.0, 1.1)

        if open_amt < 0.05:
            # Closed bud state (Stage 2): each petal forms 1/5th of a plump ovoid imbricate Sakura bud
            bud_len = 0.22 * scale * len_var
            bud_r_max = 0.086 * scale * wid_var
            env_u = np.maximum(0.0, np.sin(math.pi * (U ** 0.62))) ** 0.80
            # Slight spiral imbrication so the 5 closed petals overlap naturally in the bud
            imb_r = bud_r_max * env_u * (1.0 + 0.06 * V * np.sin(math.pi * U))
            wrap_angle = azimuth + az_jitter + V * 0.74 + 0.18 * U
            # Tilt the bud slightly so its ovoid 3D volume is clearly visible from the camera
            px = imb_r * np.cos(wrap_angle)
            pz = imb_r * np.sin(wrap_angle)
            py = bud_len * U + 0.03 * scale * (1.0 - U)
            P_bud_loc = np.stack([px, py, pz], axis=-1) @ rot_x(-0.28).T
            return center + P_bud_loc @ basis.T

        # Open / unfolding flower state
        theta_base = 0.25 + 0.38 * open_amt
        theta_mid = 0.42 + 0.88 * open_amt
        theta_tip = 0.15 + 1.08 * open_amt

        Ang = (
            theta_base * (1.0 - smoothstep(0.0, 0.50, s_norm))
            + theta_mid * smoothstep(0.0, 0.50, s_norm) * (1.0 - smoothstep(0.45, 1.0, s_norm))
            + theta_tip * smoothstep(0.45, 1.0, s_norm)
        )

        # Integrate midrib curve
        r_mid = 0.022 * scale + s_blade * np.sin(Ang * 0.82)
        y_mid = 0.005 * scale + s_blade * np.cos(Ang * 0.82)

        n_r = np.cos(Ang)
        n_y = -np.sin(Ang)

        # Delicate concave bowl cupping + central nectar furrow + marginal wave
        cup = -0.046 * scale * (1.15 - 0.35 * open_amt) * (V ** 2) * np.sin(math.pi * np.clip(s_norm, 0.0, 1.0))
        crease = 0.008 * scale * np.exp(-((V / 0.16) ** 2)) * (1.0 - 0.4 * U)
        ruffle = (
            0.007 * scale * open_amt * (abs_V ** 2.0)
            * smoothstep(0.45, 1.0, U) * np.sin(V * 3.5 * math.pi + wave_phase)
        )
        disp_n = cup + crease + ruffle

        w_scale = 0.55 + 0.45 * open_amt
        x_flat = x_blade * w_scale

        tw = roll_twist * open_amt * U
        x_tw = x_flat * np.cos(tw) - disp_n * np.sin(tw)
        dn_tw = x_flat * np.sin(tw) + disp_n * np.cos(tw)

        R_final = r_mid + dn_tw * n_r
        Y_final = y_mid + dn_tw * n_y

        az = azimuth + az_jitter
        ca, sa = math.cos(az), math.sin(az)
        px = R_final * ca - x_tw * sa
        pz = R_final * sa + x_tw * ca
        py = Y_final

        P_loc = np.stack([px, py, pz], axis=-1)
        return center + P_loc @ basis.T

    P_closed = build_petal_state(0.0)
    P_open = build_petal_state(max_openness)

    # Rich, nuanced Sakura Petal Pigmentation (RGB in [0, 1]) - prevents white blowout
    if color_tone == "somei_yoshino":
        # Soft blush pink grading from deep rose throat to delicate shell-pink tips
        c_throat = np.array([0.78, 0.22, 0.40], dtype=np.float32)
        c_mid = np.array([0.94, 0.68, 0.77], dtype=np.float32)
        c_tip = np.array([0.98, 0.87, 0.91], dtype=np.float32)
        c_vein = np.array([0.85, 0.45, 0.58], dtype=np.float32)
    elif color_tone == "shidare_pink":
        # Deeper weeping-cherry blush rose
        c_throat = np.array([0.75, 0.18, 0.36], dtype=np.float32)
        c_mid = np.array([0.92, 0.57, 0.69], dtype=np.float32)
        c_tip = np.array([0.97, 0.80, 0.86], dtype=np.float32)
        c_vein = np.array([0.82, 0.38, 0.52], dtype=np.float32)
    else:  # "ivory_blush"
        c_throat = np.array([0.80, 0.26, 0.43], dtype=np.float32)
        c_mid = np.array([0.95, 0.74, 0.82], dtype=np.float32)
        c_tip = np.array([0.99, 0.91, 0.94], dtype=np.float32)
        c_vein = np.array([0.87, 0.50, 0.63], dtype=np.float32)

    s_norm_c = np.clip(s_blade / max(1e-5, length), 0.0, 1.0)
    x_norm_c = np.clip(x_blade / max(1e-5, w_max), -1.0, 1.0)

    throat_mask = np.exp(-((s_norm_c / 0.25) ** 2)) * np.exp(-((x_norm_c / 0.58) ** 2))
    u_grad = smoothstep(0.12, 0.88, s_norm_c)[..., None]

    C_open = (1.0 - u_grad) * c_mid + u_grad * c_tip
    C_open = (1.0 - throat_mask[..., None]) * C_open + throat_mask[..., None] * c_throat

    # Delicate radiating translucent venation lines
    fan_coord = x_norm_c / (0.35 + 0.65 * s_norm_c)
    vein_pattern = (0.5 + 0.5 * np.cos(fan_coord * 7.0 * math.pi)) ** 4
    vein_weight = 0.28 * (1.0 - 0.60 * s_norm_c) * (1.0 - 0.35 * (x_norm_c ** 2))
    C_open = (1.0 - vein_weight[..., None] * vein_pattern[..., None]) * C_open + (
        vein_weight[..., None] * vein_pattern[..., None] * c_vein
    )
    # Subtle rose-blush rim along the apical notch
    notch_blush = 0.24 * smoothstep(0.70, 1.0, s_norm_c) * np.exp(-((x_norm_c / 0.22) ** 2))
    C_open = (1.0 - notch_blush[..., None]) * C_open + notch_blush[..., None] * c_vein

    # In the closed bud state (Stage 2), outer petals have a richer rose-crimson bud tone
    c_bud_base = np.array([0.74, 0.22, 0.38], dtype=np.float32)
    c_bud_top = np.array([0.92, 0.54, 0.67], dtype=np.float32)
    C_closed = (1.0 - s_norm_c[..., None]) * c_bud_base + s_norm_c[..., None] * c_bud_top

    return BotanicalElement(
        kind="petal", P=P_open, P_closed=P_closed, C=C_open, C_closed=C_closed,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        bloom_t0=bloom_t0, bloom_t1=bloom_t1,
        extra_wire_t0=bloom_t0 - 0.2, extra_wire_t1=bloom_t1,
        sss_strength=0.36, spec_strength=0.20, shininess=24.0,
        wire_color=(0.96, 0.64, 0.76), wire_stride_u=7, wire_stride_v=5,
        anchor=center, scale_t0=surf_t0, scale_t1=surf_t1 + 0.5, scale_init=0.35,
        dof_layer=dof_layer
    )


def generate_calyx_sepals(center, basis, rot_offset, scale, wire_t0, wire_t1, surf_t0, surf_t1,
                          bloom_t0, bloom_t1, dof_layer="mid"):
    """
    Generate the 5 small wine-red/green calyx sepals:
      - In Stage 2 (closed bud), they cup upward hugging the base of the pink bud.
      - In Stage 3 (blooming), they reflex outward beneath the opening petals.
    """
    sepals = []
    Nu, Nv = 14, 9
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')

    length = 0.092 * scale
    w_max = 0.034 * scale
    w_prof = w_max * (1.0 - (U ** 1.3)) * (1.0 - 0.2 * (V ** 2))

    c_base = np.array([0.36, 0.42, 0.20], dtype=np.float32)
    c_tip = np.array([0.48, 0.16, 0.22], dtype=np.float32)
    C = (1.0 - U[..., None]) * c_base + U[..., None] * c_tip

    for k in range(5):
        az = rot_offset + (k + 0.5) * (2.0 * math.pi / 5.0)
        ca, sa = math.cos(az), math.sin(az)
        x_loc = w_prof * V

        # Closed state: hugs the lower bud dome upward
        ang_closed = 0.45 - 0.18 * U
        r_c = 0.024 * scale + length * U * np.sin(ang_closed)
        y_c = 0.002 * scale + length * U * np.cos(ang_closed)
        P_c_loc = np.stack([r_c * ca - x_loc * sa, y_c, r_c * sa + x_loc * ca], axis=-1) @ rot_x(-0.28).T
        P_closed = center + P_c_loc @ basis.T

        # Open state: reflexes outward beneath the 5 petals
        ang_open = 1.08 + 0.45 * U
        r_o = 0.026 * scale + length * U * np.sin(ang_open)
        y_o = -0.004 * scale + length * U * np.cos(ang_open)
        P_o_loc = np.stack([r_o * ca - x_loc * sa, y_o, r_o * sa + x_loc * ca], axis=-1)
        P_open = center + P_o_loc @ basis.T

        sepals.append(BotanicalElement(
            kind="sepal", P=P_open, P_closed=P_closed, C=C,
            wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
            bloom_t0=bloom_t0, bloom_t1=bloom_t1,
            sss_strength=0.16, spec_strength=0.18, shininess=14.0,
            wire_color=(0.68, 0.42, 0.46), wire_stride_u=5, wire_stride_v=4,
            anchor=center, scale_t0=surf_t0, scale_t1=surf_t1, scale_init=0.4,
            dof_layer=dof_layer
        ))
    return sepals


def generate_sakura_stamens_and_pistil(center, basis, scale, wire_t0, wire_t1, surf_t0, surf_t1,
                                       bloom_t0, bloom_t1, rng_seed=500, dof_layer="mid"):
    """
    Generate the characteristic Cherry Blossom center:
      - 1 central greenish-rose pistil with stigma
      - 30 slender radiating stamen filaments arranged in two concentric whorls,
        deep wine-rose at the base transitioning to soft blush-white at the tip,
        each crowned with a delicate bilobed golden-amber pollen anther.
    """
    rng = np.random.RandomState(rng_seed)
    elements = []

    Nu, Nv = 14, 6
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    theta = (v + 1.0) * math.pi

    # 1. Central Pistil (Style + Stigma)
    p_len = 0.145 * scale
    p_pos = np.stack([np.zeros_like(u), p_len * u, np.zeros_like(u)], axis=-1)
    P_pistil = np.zeros((Nu, Nv, 3), dtype=np.float32)
    C_pistil = np.zeros((Nu, Nv, 3), dtype=np.float32)
    c_style_base = np.array([0.52, 0.22, 0.30], dtype=np.float32)
    c_style_tip = np.array([0.62, 0.68, 0.30], dtype=np.float32)
    for i in range(Nu):
        r = scale * (0.0055 if u[i] < 0.88 else 0.010)
        P_pistil[i] = p_pos[i] + np.stack([r * np.cos(theta), np.zeros_like(theta), r * np.sin(theta)], axis=-1)
        C_pistil[i] = (1.0 - u[i]) * c_style_base + u[i] * c_style_tip

    P_pistil_world = center + P_pistil @ basis.T
    elements.append(BotanicalElement(
        kind="stamen", P=P_pistil_world, C=C_pistil,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.22, spec_strength=0.25, shininess=20.0,
        wire_color=(0.75, 0.90, 0.50), wire_stride_u=6, wire_stride_v=3,
        anchor=center, dof_layer=dof_layer
    ))

    # 2. 30 Radial Stamens (18 outer whorl + 12 inner whorl)
    c_fil_base = np.array([0.74, 0.18, 0.34], dtype=np.float32)
    c_fil_tip = np.array([0.97, 0.88, 0.91], dtype=np.float32)

    whorls = [
        (18, 0.155, 0.017, 0.50, 0.68),
        (12, 0.115, 0.012, 0.30, 0.45),
    ]

    stamen_idx = 0
    for count, len_m, rb_m, ang_lo, ang_hi in whorls:
        for k in range(count):
            az = (k / float(count)) * 2.0 * math.pi + rng.uniform(-0.08, 0.08)
            s_len = len_m * scale * rng.uniform(0.90, 1.10)
            r0 = rb_m * scale
            flare_open = rng.uniform(ang_lo, ang_hi)
            flare_closed = 0.08 + 0.04 * rng.uniform(0.0, 1.0)

            ca, sa = math.cos(az), math.sin(az)
            rad_dir = np.array([ca, 0.0, sa], dtype=np.float32)
            up_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            def make_filament(flare_a):
                q0 = r0 * rad_dir + 0.005 * scale * up_dir
                q1 = (r0 + 0.28 * s_len * math.sin(flare_a * 0.7)) * rad_dir + (0.36 * s_len * math.cos(flare_a * 0.7)) * up_dir
                q2 = (r0 + 0.68 * s_len * math.sin(flare_a)) * rad_dir + (0.74 * s_len * math.cos(flare_a)) * up_dir
                q3 = (r0 + 0.96 * s_len * math.sin(flare_a * 0.92)) * rad_dir + (1.02 * s_len * math.cos(flare_a * 0.88)) * up_dir
                pos_c, _ = eval_cubic_bezier(q0, q1, q2, q3, u)
                P_f = np.zeros((Nu, Nv, 3), dtype=np.float32)
                r_fil = 0.0026 * scale
                for i in range(Nu):
                    rf = r_fil * (1.0 - 0.35 * u[i])
                    P_f[i] = pos_c[i] + np.stack([rf * np.cos(theta), np.zeros_like(theta), rf * np.sin(theta)], axis=-1)
                return center + P_f @ basis.T, center + pos_c[-1] @ basis.T

            P_open_f, tip_open = make_filament(flare_open)
            P_closed_f, tip_closed = make_filament(flare_closed)

            C_fil = np.zeros((Nu, Nv, 3), dtype=np.float32)
            for i in range(Nu):
                mix = smoothstep(0.0, 0.65, u[i])
                C_fil[i] = (1.0 - mix) * c_fil_base + mix * c_fil_tip

            # Delicate bilobed golden-yellow pollen anther at filament tip
            anther_rad = max(1.6, 2.3 * scale * rng.uniform(0.88, 1.12))
            core_bgr = (68, 210, 250)   # Warm golden pollen yellow (BGR)
            rim_bgr = (35, 138, 205)    # Richer amber-ochre shading rim (BGR)

            dly = 0.014 * stamen_idx
            elements.append(BotanicalElement(
                kind="stamen", P=P_open_f, P_closed=P_closed_f, C=C_fil,
                wire_t0=wire_t0 + dly, wire_t1=wire_t1 + dly,
                surf_t0=surf_t0 + dly, surf_t1=surf_t1 + dly,
                bloom_t0=bloom_t0 + dly, bloom_t1=bloom_t1 + dly,
                sss_strength=0.32, spec_strength=0.28, shininess=22.0,
                wire_color=(0.98, 0.84, 0.62), wire_stride_u=7, wire_stride_v=3,
                anchor=center, dof_layer=dof_layer,
                pollen_tip=(tip_closed, tip_open, anther_rad, core_bgr, rim_bgr)
            ))
            stamen_idx += 1

    return elements


def generate_serrated_sakura_leaf(origin, branch_dir, azimuth, tilt, length, width, arch_drop,
                                  copper_tint, wire_t0, wire_t1, surf_t0, surf_t1, dof_layer="mid"):
    """
    Generate a realistic 3D young Cherry Blossom leaf (Prunus serrulata) with:
      - Ovate-elliptic acuminate blade
      - Characteristic finely serrated margins (serrulate teeth along V = +-1)
      - Pinnate herringbone venation (central midrib + angled lateral veins)
      - Spring foliage gradient (emerald green to warm coppery-bronze young leaf tips)
    """
    Nu, Nv = 40, 21
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')

    base_env = (np.maximum(0.0, np.sin(math.pi * (U ** 0.66))) ** 0.80) * (1.0 - 0.14 * (U ** 2))
    serration = 0.068 * np.sin(U * 18.0 * math.pi) * smoothstep(0.10, 0.94, U) * (np.abs(V) ** 2.2)
    w_prof = width * np.maximum(0.0, base_env + serration) + 0.015 * width * ((1.0 - U) ** 3)

    def build_leaf(unfold_t):
        reach = length * (0.28 + 0.72 * unfold_t) * (0.92 * U + 0.08 * np.sin(0.5 * math.pi * U))
        elev = length * (0.42 * U - arch_drop * (0.4 + 0.6 * unfold_t) * (U ** 1.85))
        fold_angle = 1.05 * (1.0 - unfold_t) + 0.20 * unfold_t
        v_camber = fold_angle * width * (np.abs(V) ** 1.25) * (1.0 - 0.35 * U)
        x_loc = w_prof * V * (0.48 + 0.52 * unfold_t)
        y_loc = reach
        z_loc = elev + v_camber
        P_loc = np.stack([x_loc, y_loc, z_loc], axis=-1)
        rot = rot_y(azimuth) @ rot_x(tilt)
        return origin + P_loc @ rot.T

    P_closed = build_leaf(0.18)
    P_open = build_leaf(1.00)

    c_green_deep = np.array([0.15, 0.35, 0.13], dtype=np.float32)
    c_green_mid = np.array([0.27, 0.51, 0.19], dtype=np.float32)
    c_vein = np.array([0.46, 0.66, 0.27], dtype=np.float32)
    c_copper = np.array([0.52, 0.32, 0.17], dtype=np.float32)

    midrib = np.exp(-((V / 0.09) ** 2))[..., None]
    pinnate_coord = U * 7.5 - np.abs(V) * 2.8
    lat_veins = ((0.5 + 0.5 * np.cos(pinnate_coord * 2.0 * math.pi)) ** 6)[..., None] * (1.0 - midrib)

    edge_f = (np.abs(V) ** 1.5)[..., None]
    C = (1.0 - edge_f) * c_green_mid + edge_f * c_green_deep
    copper_mask = copper_tint * (0.55 * (U ** 1.6)[..., None] + 0.45 * (np.abs(V) ** 2.2)[..., None])
    C = (1.0 - copper_mask) * C + copper_mask * c_copper
    C = (1.0 - 0.52 * midrib) * C + 0.52 * midrib * c_vein
    C = C + 0.12 * lat_veins * c_vein

    return BotanicalElement(
        kind="leaf", P=P_open, P_closed=P_closed, C=C,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        bloom_t0=surf_t0, bloom_t1=surf_t1 + 1.2,
        sss_strength=0.28, spec_strength=0.26, shininess=22.0,
        wire_color=(0.56, 0.88, 0.44), wire_stride_u=6, wire_stride_v=4,
        anchor=origin, scale_t0=surf_t0, scale_t1=surf_t1 + 0.9, scale_init=0.30,
        dof_layer=dof_layer
    )


# ============================================================================
# COMPLETE SAKURA BRANCH SCENE ASSEMBLY (5 STAGES)
# ============================================================================
def build_sakura_scene():
    """
    Construct the complete 3D Japanese Cherry Blossom specimen across 5 stages:
      - Stage 1 (0.0s - 3.6s): Main gnarled bough + 5 organic secondary branches/spurs
      - Stage 2 (3.0s - 6.6s): Pedicels, calyx cups, and 23 flower buds (10 stay buds,
                               13 swell and prepare to bloom in Stage 3)
      - Stage 3 (6.2s - 10.8s): 13 five-petaled Sakura blossoms unfolding petal-by-petal
                               with bilobed apical clefts and concave cupping
      - Stage 4 (10.0s - 14.2s): 30 radial stamens & golden anthers per flower + 16 serrated leaves
      - Stage 5 (13.6s - 18.0s): Full bloom showcase + drifting 3D Sakura petals
    """
    elements = []

    # ------------------------------------------------------------------------
    # STAGE 1: BARE GNARLED WOODY BRANCHES (0.0s -> 3.6s)
    # ------------------------------------------------------------------------
    main_branch, main_pos, main_tan, _, _ = generate_woody_branch(
        p0=np.array([-0.68, -2.25, -0.05], dtype=np.float32),
        p1=np.array([-0.42, -0.95, 0.12], dtype=np.float32),
        p2=np.array([0.18, 0.25, 0.06], dtype=np.float32),
        p3=np.array([0.46, 1.52, -0.08], dtype=np.float32),
        r_base=0.105, r_tip=0.028, Nu=80, Nv=29,
        wire_t0=0.0, wire_t1=1.8, surf_t0=0.35, surf_t1=2.5,
        knot_u_list=[(0.24, 0.22), (0.48, 0.25), (0.72, 0.20)],
        rng_seed=11, dof_layer="mid"
    )
    elements.append(main_branch)

    b1_el, b1_pos, b1_tan, _, _ = generate_woody_branch(
        p0=main_pos[18],
        p1=main_pos[18] + np.array([-0.26, 0.24, 0.14], dtype=np.float32),
        p2=main_pos[18] + np.array([-0.48, 0.56, 0.22], dtype=np.float32),
        p3=main_pos[18] + np.array([-0.58, 0.92, 0.18], dtype=np.float32),
        r_base=0.064, r_tip=0.020, Nu=48, Nv=21,
        wire_t0=0.75, wire_t1=2.3, surf_t0=1.15, surf_t1=2.9,
        knot_u_list=[(0.45, 0.18)], rng_seed=22, dof_layer="mid"
    )
    elements.append(b1_el)

    b2_el, b2_pos, b2_tan, _, _ = generate_woody_branch(
        p0=main_pos[37],
        p1=main_pos[37] + np.array([0.28, 0.16, 0.16], dtype=np.float32),
        p2=main_pos[37] + np.array([0.52, 0.34, 0.20], dtype=np.float32),
        p3=main_pos[37] + np.array([0.66, 0.62, 0.12], dtype=np.float32),
        r_base=0.056, r_tip=0.018, Nu=44, Nv=19,
        wire_t0=1.05, wire_t1=2.6, surf_t0=1.45, surf_t1=3.2,
        knot_u_list=[(0.52, 0.16)], rng_seed=33, dof_layer="mid"
    )
    elements.append(b2_el)

    b3_el, b3_pos, b3_tan, _, _ = generate_woody_branch(
        p0=main_pos[55],
        p1=main_pos[55] + np.array([-0.20, 0.28, 0.10], dtype=np.float32),
        p2=main_pos[55] + np.array([-0.42, 0.56, 0.14], dtype=np.float32),
        p3=main_pos[55] + np.array([-0.54, 0.88, 0.08], dtype=np.float32),
        r_base=0.046, r_tip=0.016, Nu=42, Nv=19,
        wire_t0=1.35, wire_t1=2.85, surf_t0=1.75, surf_t1=3.45,
        knot_u_list=[(0.48, 0.15)], rng_seed=44, dof_layer="mid"
    )
    elements.append(b3_el)

    b4_el, b4_pos, b4_tan, _, _ = generate_woody_branch(
        p0=main_pos[26],
        p1=main_pos[26] + np.array([0.22, -0.04, 0.12], dtype=np.float32),
        p2=main_pos[26] + np.array([0.42, 0.02, 0.18], dtype=np.float32),
        p3=main_pos[26] + np.array([0.56, 0.16, 0.14], dtype=np.float32),
        r_base=0.044, r_tip=0.016, Nu=36, Nv=17,
        wire_t0=1.50, wire_t1=3.0, surf_t0=1.90, surf_t1=3.6,
        rng_seed=55, dof_layer="mid"
    )
    elements.append(b4_el)

    b5_el, b5_pos, b5_tan, _, _ = generate_woody_branch(
        p0=main_pos[44],
        p1=main_pos[44] + np.array([-0.15, 0.22, -0.22], dtype=np.float32),
        p2=main_pos[44] + np.array([-0.28, 0.52, -0.38], dtype=np.float32),
        p3=main_pos[44] + np.array([-0.34, 0.84, -0.46], dtype=np.float32),
        r_base=0.040, r_tip=0.015, Nu=36, Nv=15,
        wire_t0=1.65, wire_t1=3.15, surf_t0=2.05, surf_t1=3.75,
        rng_seed=66, dof_layer="bg"
    )
    elements.append(b5_el)

    # ------------------------------------------------------------------------
    # STAGE 2: UNOPENED FLOWER BUDS AT MULTIPLE DEVELOPMENT STAGES (3.1s -> 6.5s)
    # ------------------------------------------------------------------------
    buds_spec = [
        (b1_pos[42], np.array([-0.12, 0.16, 0.08]), np.array([-0.55, 0.72, 0.42]), 0.95, "cracking", 3.1, 201, "mid"),
        (b1_pos[46], np.array([-0.06, 0.18, -0.04]), np.array([-0.35, 0.88, -0.15]), 0.84, "swelling", 3.3, 202, "mid"),
        (b2_pos[38], np.array([0.16, 0.14, 0.06]), np.array([0.68, 0.62, 0.38]), 0.92, "cracking", 3.4, 203, "mid"),
        (b2_pos[42], np.array([0.14, 0.18, -0.05]), np.array([0.55, 0.80, -0.22]), 0.78, "tight", 3.6, 204, "mid"),
        (b3_pos[36], np.array([-0.14, 0.18, 0.06]), np.array([-0.58, 0.76, 0.28]), 0.90, "swelling", 3.7, 205, "mid"),
        (b3_pos[40], np.array([-0.08, 0.20, -0.02]), np.array([-0.32, 0.92, -0.12]), 0.76, "tight", 3.9, 206, "mid"),
        (main_pos[73], np.array([0.10, 0.20, 0.04]), np.array([0.38, 0.88, 0.25]), 0.94, "cracking", 4.0, 207, "mid"),
        (main_pos[77], np.array([0.06, 0.22, -0.04]), np.array([0.24, 0.95, -0.18]), 0.80, "swelling", 4.2, 208, "mid"),
        (b4_pos[33], np.array([0.16, 0.06, 0.08]), np.array([0.78, 0.35, 0.52]), 0.86, "swelling", 4.3, 209, "mid"),
        (b5_pos[33], np.array([-0.10, 0.18, -0.08]), np.array([-0.42, 0.82, -0.38]), 0.85, "cracking", 4.4, 210, "bg"),
    ]

    for b_orig, b_off, b_dir, b_sc, b_stage, b_t0, b_seed, b_dof in buds_spec:
        b_orig = np.asarray(b_orig, dtype=np.float32)
        b_off = np.asarray(b_off, dtype=np.float32)
        b_dir = normalize_v3(np.asarray(b_dir, dtype=np.float32))
        q0 = b_orig
        q3 = b_orig + b_off
        q1 = q0 + 0.35 * b_off + np.array([0.0, 0.03, 0.02], dtype=np.float32)
        q2 = q0 + 0.70 * b_off - 0.04 * b_dir

        ped_el, tip_p, _ = generate_pedicel_and_calyx(
            q0, q1, q2, q3, scale=b_sc,
            wire_t0=b_t0, wire_t1=b_t0 + 1.0,
            surf_t0=b_t0 + 0.25, surf_t1=b_t0 + 1.35,
            dof_layer=b_dof
        )
        elements.append(ped_el)
        elements.append(generate_sakura_bud(
            tip_pos=tip_p, tip_dir=b_dir, scale=b_sc, stage=b_stage,
            wire_t0=b_t0 + 0.45, wire_t1=b_t0 + 1.6,
            surf_t0=b_t0 + 0.75, surf_t1=b_t0 + 2.1,
            rng_seed=b_seed, dof_layer=b_dof
        ))

    # ------------------------------------------------------------------------
    # STAGE 2 -> 3 -> 4: THIRTEEN BLOOMING SAKURA FLOWERS IN CORYMB CLUSTERS
    # ------------------------------------------------------------------------
    # In Stage 2, each flower's pedicel, calyx, and closed bud petals form a
    # plump rose-pink bud. In Stage 3 (6.2s -> 10.8s), the 5 petals unfold
    # individually with staggered timing into a full 5-petal Sakura blossom!
    flowers_spec = [
        # 1. Hero Center Blossom (Main focal point, facing camera-front-right)
        {
            "node": main_pos[40],
            "offset": np.array([-0.04, 0.14, 0.26]),
            "dir": np.array([0.08, 0.22, 0.97]),
            "scale": 1.15, "open": 1.00, "tone": "somei_yoshino", "rot": 0.18,
            "ped_t0": 3.2, "petal_t0": 6.2, "stamen_t0": 10.0, "seed": 301, "dof": "mid"
        },
        # 2. Hero Center-Left Companion Blossom
        {
            "node": main_pos[36],
            "offset": np.array([-0.26, -0.02, 0.22]),
            "dir": np.array([-0.36, 0.18, 0.91]),
            "scale": 1.04, "open": 0.96, "tone": "shidare_pink", "rot": 0.82,
            "ped_t0": 3.4, "petal_t0": 6.5, "stamen_t0": 10.2, "seed": 310, "dof": "mid"
        },
        # 3. Hero Lower-Left Branch Blossom
        {
            "node": b1_pos[24],
            "offset": np.array([-0.14, 0.12, 0.22]),
            "dir": np.array([-0.28, 0.28, 0.92]),
            "scale": 1.08, "open": 0.98, "tone": "shidare_pink", "rot": 0.45,
            "ped_t0": 3.5, "petal_t0": 6.8, "stamen_t0": 10.4, "seed": 320, "dof": "mid"
        },
        # 4. Lower-Left Outer Companion Blossom (3/4 profile view)
        {
            "node": b1_pos[34],
            "offset": np.array([-0.18, 0.08, 0.16]),
            "dir": np.array([-0.55, 0.34, 0.76]),
            "scale": 0.94, "open": 0.90, "tone": "somei_yoshino", "rot": 1.10,
            "ped_t0": 3.7, "petal_t0": 7.1, "stamen_t0": 10.6, "seed": 330, "dof": "mid"
        },
        # 5. Hero Mid-Right Branch Blossom
        {
            "node": b2_pos[20],
            "offset": np.array([0.12, 0.14, 0.22]),
            "dir": np.array([0.32, 0.26, 0.91]),
            "scale": 1.10, "open": 1.00, "tone": "ivory_blush", "rot": -0.25,
            "ped_t0": 3.6, "petal_t0": 7.0, "stamen_t0": 10.5, "seed": 340, "dof": "mid"
        },
        # 6. Mid-Right Outer Blossom
        {
            "node": b2_pos[31],
            "offset": np.array([0.18, 0.06, 0.16]),
            "dir": np.array([0.58, 0.22, 0.78]),
            "scale": 0.95, "open": 0.92, "tone": "somei_yoshino", "rot": 0.60,
            "ped_t0": 3.8, "petal_t0": 7.3, "stamen_t0": 10.8, "seed": 350, "dof": "mid"
        },
        # 7. Upper-Center Main Bough Hero Blossom
        {
            "node": main_pos[54],
            "offset": np.array([0.08, 0.12, 0.22]),
            "dir": np.array([0.18, 0.32, 0.93]),
            "scale": 1.06, "open": 0.96, "tone": "somei_yoshino", "rot": 0.35,
            "ped_t0": 3.9, "petal_t0": 7.5, "stamen_t0": 10.9, "seed": 360, "dof": "mid"
        },
        # 8. Upper-Left Crown Blossom
        {
            "node": b3_pos[24],
            "offset": np.array([-0.12, 0.14, 0.20]),
            "dir": np.array([-0.30, 0.36, 0.88]),
            "scale": 1.02, "open": 0.95, "tone": "shidare_pink", "rot": 0.95,
            "ped_t0": 4.0, "petal_t0": 7.7, "stamen_t0": 11.1, "seed": 370, "dof": "mid"
        },
        # 9. Lower-Center Trunk Spur Blossom
        {
            "node": main_pos[24],
            "offset": np.array([0.06, -0.06, 0.24]),
            "dir": np.array([0.16, -0.14, 0.98]),
            "scale": 1.02, "open": 0.96, "tone": "ivory_blush", "rot": 0.52,
            "ped_t0": 3.5, "petal_t0": 6.7, "stamen_t0": 10.3, "seed": 380, "dof": "mid"
        },
        # 10. Lower-Right Spur Blossom
        {
            "node": b4_pos[22],
            "offset": np.array([0.14, -0.04, 0.20]),
            "dir": np.array([0.46, -0.10, 0.88]),
            "scale": 0.94, "open": 0.88, "tone": "somei_yoshino", "rot": -0.40,
            "ped_t0": 4.1, "petal_t0": 7.9, "stamen_t0": 11.3, "seed": 390, "dof": "mid"
        },
        # 11. Upper-Right Crown Blossom
        {
            "node": main_pos[65],
            "offset": np.array([0.14, 0.10, 0.18]),
            "dir": np.array([0.38, 0.35, 0.86]),
            "scale": 0.94, "open": 0.90, "tone": "ivory_blush", "rot": 0.12,
            "ped_t0": 4.2, "petal_t0": 8.1, "stamen_t0": 11.4, "seed": 400, "dof": "mid"
        },
        # 12. Background Depth Blossom 1 (Soft out-of-focus upper-left depth)
        {
            "node": b5_pos[18],
            "offset": np.array([-0.10, 0.12, -0.10]),
            "dir": np.array([-0.25, 0.28, 0.92]),
            "scale": 0.95, "open": 0.94, "tone": "shidare_pink", "rot": 0.30,
            "ped_t0": 4.1, "petal_t0": 7.8, "stamen_t0": 11.2, "seed": 410, "dof": "bg"
        },
        # 13. Background Depth Blossom 2 (Soft out-of-focus mid-center depth)
        {
            "node": b5_pos[27],
            "offset": np.array([0.08, 0.14, -0.12]),
            "dir": np.array([0.18, 0.32, 0.93]),
            "scale": 0.88, "open": 0.90, "tone": "somei_yoshino", "rot": 0.75,
            "ped_t0": 4.3, "petal_t0": 8.0, "stamen_t0": 11.4, "seed": 420, "dof": "bg"
        },
    ]

    for f in flowers_spec:
        node = np.asarray(f["node"], dtype=np.float32)
        off = np.asarray(f["offset"], dtype=np.float32)
        f_dir = normalize_v3(np.asarray(f["dir"], dtype=np.float32))
        sc = f["scale"]
        dof = f["dof"]

        q0 = node
        q3 = node + off
        q1 = q0 + 0.36 * off + np.array([0.0, 0.04, 0.02], dtype=np.float32)
        q2 = q0 + 0.72 * off - 0.05 * f_dir

        ped_t0 = f["ped_t0"]
        ped_el, flower_center, _ = generate_pedicel_and_calyx(
            q0, q1, q2, q3, scale=sc,
            wire_t0=ped_t0, wire_t1=ped_t0 + 1.1,
            surf_t0=ped_t0 + 0.3, surf_t1=ped_t0 + 1.5,
            dof_layer=dof
        )
        elements.append(ped_el)

        basis = build_basis_from_dir(f_dir)

        pt0 = f["petal_t0"]
        elements.extend(generate_calyx_sepals(
            center=flower_center, basis=basis, rot_offset=f["rot"], scale=sc,
            wire_t0=ped_t0 + 0.55, wire_t1=ped_t0 + 1.35,
            surf_t0=ped_t0 + 0.75, surf_t1=ped_t0 + 1.55,
            bloom_t0=pt0, bloom_t1=pt0 + 2.0,
            dof_layer=dof
        ))

        # Petals form a closed rose-pink bud in Stage 2 (ped_t0 + 0.6 -> ped_t0 + 1.7)
        # and then individually unfold & expand in Stage 3 (pt0 + p_delay -> pt0 + p_delay + 2.5)
        bud_form_t0 = ped_t0 + 0.55
        for k in range(5):
            az = f["rot"] + k * (2.0 * math.pi / 5.0)
            p_delay = 0.25 * k
            elements.append(generate_sakura_petal(
                center=flower_center, basis=basis, azimuth=az,
                scale=sc, max_openness=f["open"], color_tone=f["tone"],
                wire_t0=bud_form_t0 + 0.05 * k,
                wire_t1=bud_form_t0 + 0.95 + 0.05 * k,
                surf_t0=bud_form_t0 + 0.20 + 0.05 * k,
                surf_t1=bud_form_t0 + 1.25 + 0.05 * k,
                bloom_t0=pt0 + p_delay,
                bloom_t1=pt0 + p_delay + 2.5,
                rng_seed=f["seed"] + k,
                dof_layer=dof
            ))

        st0 = f["stamen_t0"]
        elements.extend(generate_sakura_stamens_and_pistil(
            center=flower_center, basis=basis, scale=sc,
            wire_t0=st0 - 0.3, wire_t1=st0 + 1.0,
            surf_t0=st0, surf_t1=st0 + 1.5,
            bloom_t0=st0 + 0.1, bloom_t1=st0 + 2.0,
            rng_seed=f["seed"] + 50,
            dof_layer=dof
        ))

    # ------------------------------------------------------------------------
    # STAGE 4: SERRATED YOUNG GREEN SAKURA LEAVES (10.0s -> 13.8s)
    # ------------------------------------------------------------------------
    leaves_spec = [
        (main_pos[14], main_tan[14],  0.78,  0.32, 0.44, 0.17, 0.24, 0.32, 10.0, "mid"),
        (main_pos[21], main_tan[21], -1.18,  0.28, 0.48, 0.18, 0.26, 0.24, 10.2, "mid"),
        (main_pos[31], main_tan[31],  0.95,  0.35, 0.46, 0.17, 0.25, 0.28, 10.4, "mid"),
        (main_pos[47], main_tan[47], -1.35,  0.30, 0.44, 0.16, 0.22, 0.38, 10.6, "mid"),
        (main_pos[61], main_tan[61],  0.82,  0.26, 0.40, 0.15, 0.20, 0.50, 10.8, "mid"),
        (main_pos[70], main_tan[70], -0.92,  0.22, 0.35, 0.13, 0.18, 0.60, 11.0, "mid"),

        (b1_pos[14],   b1_tan[14],   -1.85,  0.30, 0.44, 0.16, 0.24, 0.28, 10.3, "mid"),
        (b1_pos[29],   b1_tan[29],    0.45,  0.25, 0.40, 0.15, 0.22, 0.42, 10.5, "mid"),
        (b1_pos[39],   b1_tan[39],   -1.45,  0.22, 0.34, 0.13, 0.18, 0.56, 10.9, "mid"),

        (b2_pos[13],   b2_tan[13],    1.25,  0.32, 0.44, 0.16, 0.24, 0.28, 10.4, "mid"),
        (b2_pos[27],   b2_tan[27],   -0.55,  0.28, 0.40, 0.15, 0.20, 0.44, 10.7, "mid"),
        (b2_pos[36],   b2_tan[36],    1.55,  0.24, 0.34, 0.13, 0.18, 0.58, 11.0, "mid"),

        (b3_pos[16],   b3_tan[16],   -1.50,  0.28, 0.40, 0.15, 0.22, 0.38, 10.8, "mid"),
        (b3_pos[31],   b3_tan[31],    0.65,  0.24, 0.35, 0.13, 0.18, 0.54, 11.1, "mid"),

        (b4_pos[15],   b4_tan[15],    1.40,  0.34, 0.38, 0.14, 0.22, 0.35, 10.6, "mid"),
        (b4_pos[29],   b4_tan[29],   -0.65,  0.26, 0.34, 0.13, 0.18, 0.52, 11.0, "mid"),
    ]

    for l_orig, l_tan, l_az, l_tilt, l_len, l_wid, l_arch, l_cop, l_t0, l_dof in leaves_spec:
        elements.append(generate_serrated_sakura_leaf(
            origin=np.asarray(l_orig, dtype=np.float32),
            branch_dir=np.asarray(l_tan, dtype=np.float32),
            azimuth=l_az, tilt=l_tilt, length=l_len, width=l_wid,
            arch_drop=l_arch, copper_tint=l_cop,
            wire_t0=l_t0 - 0.3, wire_t1=l_t0 + 1.0,
            surf_t0=l_t0, surf_t1=l_t0 + 1.6,
            dof_layer=l_dof
        ))

    return elements


# ============================================================================
# 3D DRIFTING SAKURA PETAL PARTICLE SYSTEM (STAGE 4 -> STAGE 5)
# ============================================================================
def build_drifting_petals_template():
    """
    Create a canonical 3D notched Sakura petal mesh + trajectories for 22
    floating petals that gently drift, tumble, and catch the light in Stage 5.
    """
    Nu, Nv = 20, 15
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')
    abs_V = np.abs(V)

    length = 0.22
    w_max = 0.095
    theta_p = V * (0.84 * math.pi)
    s_oval = 0.50 + 0.50 * np.cos(theta_p)
    lobe_boost = 0.065 * np.exp(-(((abs_V - 0.23) / 0.15) ** 2))
    cleft_notch = 0.175 * (np.maximum(0.0, 1.0 - (abs_V / 0.17)) ** 1.45)
    S_edge = length * np.maximum(0.02, s_oval + lobe_boost - cleft_notch)
    X_edge = w_max * np.sin(theta_p) * (1.0 + 0.14 * np.cos(theta_p))

    u_r = U ** 0.88
    s_blade = (1.0 - u_r) * (0.015 * length * (1.0 - V ** 2)) + u_r * S_edge
    x_loc = (1.0 - u_r) * (0.09 * w_max * V) + u_r * X_edge
    y_loc = s_blade - 0.5 * length
    z_loc = -0.022 * ((x_loc / w_max) ** 2) + 0.014 * (((s_blade / length) - 0.5) ** 2)
    P_base = np.stack([x_loc, y_loc, z_loc], axis=-1).astype(np.float32)

    c_base = np.array([0.92, 0.62, 0.73], dtype=np.float32)
    c_tip = np.array([0.98, 0.88, 0.92], dtype=np.float32)
    C_base = (1.0 - U[..., None]) * c_base + U[..., None] * c_tip

    rng = np.random.RandomState(777)
    drifters = []
    for i in range(22):
        t_spawn = rng.uniform(11.6, 14.2)
        start_pos = np.array([
            rng.uniform(-0.75, 0.85),
            rng.uniform(0.35, 1.75),
            rng.uniform(-0.45, 0.55)
        ], dtype=np.float32)
        vel = np.array([
            rng.uniform(0.08, 0.22) * (1 if i % 2 == 0 else -0.8),
            rng.uniform(-0.26, -0.14),
            rng.uniform(-0.06, 0.08)
        ], dtype=np.float32)
        rot_rates = rng.uniform(1.1, 2.4, size=3).astype(np.float32)
        rot_phases = rng.uniform(0.0, 2.0 * math.pi, size=3).astype(np.float32)
        scale_p = rng.uniform(0.75, 1.18)
        if start_pos[2] > 0.35:
            dof = "fg"
        elif start_pos[2] < -0.22:
            dof = "bg"
        else:
            dof = "mid"
        drifters.append({
            "t_spawn": t_spawn,
            "start_pos": start_pos,
            "vel": vel,
            "rot_rates": rot_rates,
            "rot_phases": rot_phases,
            "scale": scale_p,
            "dof": dof,
        })
    return P_base, C_base, drifters


DRIFT_P_BASE, DRIFT_C_BASE, DRIFT_SPECS = build_drifting_petals_template()


# ============================================================================
# STUDIO BACKDROP, CAMERA PROJECTION & PBR/SSS SHADING
# ============================================================================
def create_sakura_studio_backdrop():
    """
    Precompute a dark, warm, cinematic 9:16 botanical studio backdrop with
    subtle twilight plum-amber radial atmospheric glow.
    """
    y = np.linspace(-1.0, 1.0, HEIGHT, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32)[None, :]

    r2_center = (x * 0.90) ** 2 + ((y + 0.05) * 0.72) ** 2
    glow_main = np.exp(-1.55 * r2_center)

    r2_warm = ((x - 0.22) * 0.85) ** 2 + ((y + 0.28) * 0.80) ** 2
    glow_warm = np.exp(-2.1 * r2_warm)

    b = 12.0 + 20.0 * glow_main + 12.0 * glow_warm
    g = 10.0 + 17.0 * glow_main + 13.0 * glow_warm
    r = 15.0 + 26.0 * glow_main + 24.0 * glow_warm

    bg = np.stack([b, g, r], axis=-1)
    return np.clip(bg, 0, 255).astype(np.uint8)


STUDIO_BG = create_sakura_studio_backdrop()


def project_points(P_world, cam_rot, cam_target, cam_dist, fov_scale):
    """Transform 3D world points into camera space and 2D screen coordinates."""
    P_rel = (P_world - cam_target) @ cam_rot.T
    z_depth = cam_dist - P_rel[..., 2]
    z_safe = np.maximum(z_depth, 0.25)
    sx = (WIDTH * 0.50) + (P_rel[..., 0] / z_safe) * fov_scale
    sy = (HEIGHT * 0.52) - (P_rel[..., 1] / z_safe) * fov_scale
    return sx, sy, z_depth, P_rel


def compute_lighting(normals_cam, base_rgb, sss_strength, spec_strength, shininess):
    """
    Physically-based 3D botanical shader:
      - Warm golden-hour key light from upper-left-front
      - Soft cool-lavender sky fill light from right
      - Rim backlight + Subsurface Scattering (SSS) for translucent Sakura petals
      - Micro-cuticle Blinn-Phong specular highlight
      - Natural filmic tone mapping that preserves delicate pink gradients
    """
    L_key = normalize_v3(np.array([-0.38, 0.58, 0.72], dtype=np.float32))
    L_fill = normalize_v3(np.array([0.68, -0.18, 0.71], dtype=np.float32))
    L_rim = normalize_v3(np.array([0.15, 0.75, -0.64], dtype=np.float32))
    V_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    nz = normals_cam[..., 2:3]
    N = np.where(nz < 0.0, -normals_cam, normals_cam)

    ndotl_key = np.sum(N * L_key, axis=-1, keepdims=True)
    diff_key = np.clip((ndotl_key + 0.22) / 1.22, 0.0, 1.0)

    ndotl_fill = np.sum(N * L_fill, axis=-1, keepdims=True)
    diff_fill = np.clip((ndotl_fill + 0.35) / 1.35, 0.0, 1.0)

    ndotv = np.clip(np.sum(N * V_dir, axis=-1, keepdims=True), 0.0, 1.0)
    fresnel_rim = (1.0 - ndotv) ** 2.2
    back_trans = np.clip(np.sum(N * L_rim, axis=-1, keepdims=True) * 0.5 + 0.5, 0.0, 1.0)
    sss = sss_strength * (0.55 * fresnel_rim + 0.45 * back_trans)

    H_key = normalize_v3(L_key + V_dir)
    ndoth = np.clip(np.sum(N * H_key, axis=-1, keepdims=True), 0.0, 1.0)
    spec = spec_strength * (ndoth ** shininess)

    key_col = np.array([1.02, 0.98, 0.95], dtype=np.float32)
    fill_col = np.array([0.78, 0.82, 0.92], dtype=np.float32)
    sss_col = np.array([1.02, 0.74, 0.84], dtype=np.float32)

    lit_rgb = (
        base_rgb * (0.18 + 0.68 * diff_key * key_col + 0.18 * diff_fill * fill_col)
        + base_rgb * sss * sss_col
        + spec * np.array([1.0, 0.96, 0.94], dtype=np.float32)
    )

    lit_rgb = np.clip(lit_rgb, 0.0, 1.25)
    lit_rgb = lit_rgb / (1.0 + 0.20 * lit_rgb) * 1.12
    return np.clip(lit_rgb, 0.0, 1.0)


# ============================================================================
# SINGLE-FRAME 3D RENDERER (WITH DEPTH OF FIELD & BOKEH)
# ============================================================================
SCENE_ELEMENTS = None


def init_worker(w=None, h=None, show_hud=False):
    global SCENE_ELEMENTS, WIDTH, HEIGHT, STUDIO_BG, SHOW_HUD
    if w is not None and h is not None:
        WIDTH, HEIGHT = w, h
        STUDIO_BG = create_sakura_studio_backdrop()
    SHOW_HUD = show_hud
    SCENE_ELEMENTS = build_sakura_scene()


def rasterize_layer(canvas, quad_pts_list, quad_z_list, quad_bgr_list, pollen_list):
    """Depth-sort and rasterize 3D surface quads + golden pollen anthers onto `canvas`."""
    if not quad_pts_list and not pollen_list:
        return

    pollen_list.sort(key=lambda x: x[0], reverse=True)
    n_pol = len(pollen_list)
    pol_ptr = 0

    if quad_pts_list:
        all_pts = np.concatenate(quad_pts_list, axis=0)
        all_z = np.concatenate(quad_z_list, axis=0)
        all_bgr = np.concatenate(quad_bgr_list, axis=0).tolist()
        order = np.argsort(all_z)[::-1]

        for idx in order:
            qz = float(all_z[idx])
            while pol_ptr < n_pol and pollen_list[pol_ptr][0] >= qz:
                _, px_i, py_i, r_int, core_bgr, rim_bgr = pollen_list[pol_ptr]
                cv2.circle(canvas, (px_i, py_i), r_int + 1, rim_bgr, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (px_i, py_i), r_int, core_bgr, -1, lineType=cv2.LINE_AA)
                pol_ptr += 1

            cv2.fillConvexPoly(
                canvas, all_pts[idx], all_bgr[idx], lineType=cv2.LINE_AA, shift=SHIFT
            )

    while pol_ptr < n_pol:
        _, px_i, py_i, r_int, core_bgr, rim_bgr = pollen_list[pol_ptr]
        cv2.circle(canvas, (px_i, py_i), r_int + 1, rim_bgr, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (px_i, py_i), r_int, core_bgr, -1, lineType=cv2.LINE_AA)
        pol_ptr += 1


def render_frame(frame_idx, total_frames=TOTAL_FRAMES):
    """Render frame `frame_idx` at 1080x1920 resolution and return BGR uint8 image."""
    global SCENE_ELEMENTS
    if SCENE_ELEMENTS is None:
        SCENE_ELEMENTS = build_sakura_scene()

    t = (frame_idx / max(1, total_frames - 1)) * DURATION
    p_all = frame_idx / max(1, total_frames - 1)

    bg_fade = 0.35 + 0.65 * smootherstep(0.0, 1.5, t)
    canvas = (STUDIO_BG.astype(np.float32) * bg_fade).astype(np.uint8)

    # ------------------------------------------------------------------------
    # 1. Cinematic 3D Camera Trajectory (Follows growth upward, then orbits)
    # ------------------------------------------------------------------------
    azimuth = (
        -0.24
        + 0.44 * smootherstep(0.0, 0.72, p_all)
        + 0.16 * math.sin(max(0.0, (p_all - 0.65) / 0.35) * math.pi)
    )
    elevation = (
        0.04
        + 0.10 * smootherstep(0.12, 0.65, p_all)
        - 0.04 * smootherstep(0.72, 1.0, p_all)
    )
    cam_rot = rot_x(elevation) @ rot_y(azimuth)

    y_target = -0.42 * (1.0 - smootherstep(0.0, 0.35, p_all)) - 0.04 * smootherstep(0.18, 0.65, p_all)
    x_target = -0.10 * (1.0 - smootherstep(0.0, 0.28, p_all)) + 0.02 * smootherstep(0.28, 0.80, p_all)
    cam_target = np.array([x_target, y_target, 0.06], dtype=np.float32)
    cam_dist = 4.80 - 0.10 * smootherstep(0.20, 0.88, p_all)
    fov_scale = 1580.0 * (HEIGHT / 1920.0)

    wind_amp = 0.022 * smootherstep(6.5, 14.0, t)
    wind_vec = np.array([
        wind_amp * math.sin(t * 1.45),
        wind_amp * 0.45 * math.cos(t * 1.15),
        wind_amp * 0.60 * math.sin(t * 1.25 + 0.8)
    ], dtype=np.float32)

    # ------------------------------------------------------------------------
    # 2. Evaluate & Bucket Active 3D Elements by Depth-of-Field Layer
    # ------------------------------------------------------------------------
    layers = {
        "bg": {"pts": [], "z": [], "bgr": [], "pollen": []},
        "mid": {"pts": [], "z": [], "bgr": [], "pollen": []},
        "fg": {"pts": [], "z": [], "bgr": [], "pollen": []},
    }
    wire_lines = []
    wave_particles = []

    global_wire_alpha = 1.0 - smootherstep(13.0, 14.5, t)

    for el in SCENE_ELEMENTS:
        if t < el.wire_t0:
            continue

        w_prog = smootherstep(el.wire_t0, el.wire_t1, t)
        s_prog = smootherstep(el.surf_t0, el.surf_t1, t)

        P_full, C_full, b_prog = el.evaluate_mesh(t, wind_vec=wind_vec)

        # A. Solid PBR/SSS Surface Quads (smoothly sliced along u)
        if s_prog > 0.002:
            P_sub, normals, c_quad, u_quad = slice_mesh_along_u(P_full, C_full, s_prog)
            normals_cam = normals @ cam_rot.T
            sx, sy, z_depth, _ = project_points(P_sub, cam_rot, cam_target, cam_dist, fov_scale)

            lit_rgb = compute_lighting(
                normals_cam, c_quad, el.sss_strength, el.spec_strength, el.shininess
            )

            if s_prog < 0.992:
                dist_edge = s_prog - u_quad
                rim_glow = np.exp(-((dist_edge / 0.075) ** 2)) * (dist_edge >= -0.02)
                glow_col = np.array([1.0, 0.82, 0.68], dtype=np.float32)
                lit_rgb = np.clip(lit_rgb + 0.45 * rim_glow[..., None] * glow_col, 0.0, 1.0)

                if s_prog > 0.04:
                    tip_row = P_sub.shape[0] - 1
                    for j_idx in range(0, P_sub.shape[1], 4):
                        wave_particles.append((
                            float(sx[tip_row, j_idx]),
                            float(sy[tip_row, j_idx]),
                            (185, 215, 255),
                            2.2
                        ))

            q_x = np.stack([sx[:-1, :-1], sx[1:, :-1], sx[1:, 1:], sx[:-1, 1:]], axis=-1)
            q_y = np.stack([sy[:-1, :-1], sy[1:, :-1], sy[1:, 1:], sy[:-1, 1:]], axis=-1)
            cx_q = np.mean(q_x, axis=-1, keepdims=True)
            cy_q = np.mean(q_y, axis=-1, keepdims=True)
            dx_q = q_x - cx_q
            dy_q = q_y - cy_q
            d_len = np.maximum(np.hypot(dx_q, dy_q), 1e-4)
            q_x_exp = q_x + (dx_q / d_len) * 0.55
            q_y_exp = q_y + (dy_q / d_len) * 0.55

            pts_sub = np.stack([q_x_exp, q_y_exp], axis=-1).reshape(-1, 4, 2)
            pts_int = np.rint(pts_sub * SUBPIX).astype(np.int32)
            z_q = (0.25 * (z_depth[:-1, :-1] + z_depth[1:, :-1] + z_depth[1:, 1:] + z_depth[:-1, 1:])).reshape(-1)
            bgr_q = np.rint(lit_rgb[..., ::-1] * 255.0).astype(np.uint8).reshape(-1, 3)

            bucket = layers[el.dof_layer]
            bucket["pts"].append(pts_int)
            bucket["z"].append(z_q)
            bucket["bgr"].append(bgr_q)

            if el.pollen_tip is not None and s_prog > 0.88:
                tip_c, tip_o, rad_px, core_bgr, rim_bgr = el.pollen_tip
                tip_3d = tip_c * (1.0 - b_prog) + tip_o * b_prog
                if el.anchor is not None:
                    dist_a = np.linalg.norm(tip_3d - el.anchor)
                    tip_3d = tip_3d + (dist_a ** 1.35) * 0.22 * wind_vec
                tsx, tsy, tzd, _ = project_points(tip_3d[None, :], cam_rot, cam_target, cam_dist, fov_scale)
                anther_scale = smootherstep(0.88, 1.0, s_prog)
                r_int = max(1, int(round(rad_px * anther_scale)))
                bucket["pollen"].append((
                    float(tzd[0]) - 0.012,
                    int(round(float(tsx[0]))),
                    int(round(float(tsy[0]))),
                    r_int, core_bgr, rim_bgr
                ))

        # B. Parametric 3D Construction Wireframe Curves
        base_wire_vis = 1.0 - 0.88 * smootherstep(0.55, 1.0, s_prog)
        if el.extra_wire_t0 is not None and el.extra_wire_t1 is not None:
            # Re-illuminate petal wireframe curves while the petal is actively unfolding in Stage 3
            unfold_pulse = math.sin(smootherstep(el.extra_wire_t0, el.extra_wire_t1, t) * math.pi)
            base_wire_vis = max(base_wire_vis, 0.58 * unfold_pulse)

        wire_vis = global_wire_alpha * base_wire_vis
        if w_prog > 0.01 and wire_vis > 0.04:
            P_w, _, _, _ = slice_mesh_along_u(P_full, C_full, w_prog)
            wsx, wsy, wzd, _ = project_points(P_w, cam_rot, cam_target, cam_dist, fov_scale)
            Nu_w, Nv_w = P_w.shape[:2]
            wc_bgr = tuple(int(round(c * 255.0 * (0.30 + 0.70 * wire_vis))) for c in el.wire_color[::-1])

            v_indices = list(range(0, Nv_w, el.wire_stride_v))
            if (Nv_w - 1) not in v_indices:
                v_indices.append(Nv_w - 1)
            for j in v_indices:
                pts_2d = np.stack([wsx[:, j], wsy[:, j]], axis=-1)
                pts_sub = np.rint(pts_2d * SUBPIX).astype(np.int32)
                z_m = float(np.mean(wzd[:, j])) - 0.015
                wire_lines.append((pts_sub, z_m, wc_bgr, 1))
                if w_prog < 0.97 and (j % (el.wire_stride_v * 2) == 0):
                    wave_particles.append((
                        float(wsx[-1, j]), float(wsy[-1, j]), (200, 230, 255), 2.5
                    ))

            for i in range(0, Nu_w, el.wire_stride_u):
                pts_2d = np.stack([wsx[i, :], wsy[i, :]], axis=-1)
                pts_sub = np.rint(pts_2d * SUBPIX).astype(np.int32)
                z_m = float(np.mean(wzd[i, :])) - 0.015
                wire_lines.append((pts_sub, z_m, wc_bgr, 1))

    # ------------------------------------------------------------------------
    # 3. Evaluate 3D Drifting Sakura Petals (Stage 4 -> Stage 5)
    # ------------------------------------------------------------------------
    for d in DRIFT_SPECS:
        dt = t - d["t_spawn"]
        if dt <= 0.0:
            continue
        fade_in = smootherstep(0.0, 0.8, dt)
        sc = d["scale"] * fade_in
        if sc < 0.05:
            continue

        pos_c = (
            d["start_pos"]
            + d["vel"] * dt
            + np.array([
                0.11 * math.sin(dt * 1.9 + d["rot_phases"][0]),
                0.04 * math.cos(dt * 2.4 + d["rot_phases"][1]),
                0.08 * math.sin(dt * 1.6 + d["rot_phases"][2])
            ], dtype=np.float32)
        )
        pos_c[1] = ((pos_c[1] + 1.85) % 3.7) - 1.85

        rx = dt * d["rot_rates"][0] + d["rot_phases"][0]
        ry = dt * d["rot_rates"][1] + d["rot_phases"][1]
        rz = dt * d["rot_rates"][2] + d["rot_phases"][2]
        R_pet = rot_z(rz) @ rot_y(ry) @ rot_x(rx)

        P_d = pos_c + (DRIFT_P_BASE * sc) @ R_pet.T
        _, normals_d, c_quad_d, _ = slice_mesh_along_u(P_d, DRIFT_C_BASE, 1.0)
        normals_cam_d = normals_d @ cam_rot.T
        sx_d, sy_d, zd_d, _ = project_points(P_d, cam_rot, cam_target, cam_dist, fov_scale)

        lit_d = compute_lighting(normals_cam_d, c_quad_d, 0.42, 0.22, 22.0)
        q_x = np.stack([sx_d[:-1, :-1], sx_d[1:, :-1], sx_d[1:, 1:], sx_d[:-1, 1:]], axis=-1)
        q_y = np.stack([sy_d[:-1, :-1], sy_d[1:, :-1], sy_d[1:, 1:], sy_d[:-1, 1:]], axis=-1)
        pts_int = np.rint(np.stack([q_x, q_y], axis=-1).reshape(-1, 4, 2) * SUBPIX).astype(np.int32)
        z_q = (0.25 * (zd_d[:-1, :-1] + zd_d[1:, :-1] + zd_d[1:, 1:] + zd_d[:-1, 1:])).reshape(-1)
        bgr_q = np.rint(lit_d[..., ::-1] * 255.0).astype(np.uint8).reshape(-1, 3)

        bucket = layers[d["dof"]]
        bucket["pts"].append(pts_int)
        bucket["z"].append(z_q)
        bucket["bgr"].append(bgr_q)

    # ------------------------------------------------------------------------
    # 4. Composite Depth-of-Field Layers (Background Blur -> Mid Focus -> Foreground Bokeh)
    # ------------------------------------------------------------------------
    if layers["bg"]["pts"] or layers["bg"]["pollen"]:
        rasterize_layer(
            canvas,
            layers["bg"]["pts"], layers["bg"]["z"], layers["bg"]["bgr"],
            layers["bg"]["pollen"]
        )
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=3.2)

    rasterize_layer(
        canvas,
        layers["mid"]["pts"], layers["mid"]["z"], layers["mid"]["bgr"],
        layers["mid"]["pollen"]
    )

    if layers["fg"]["pts"]:
        fg_canvas = canvas.copy()
        rasterize_layer(
            fg_canvas,
            layers["fg"]["pts"], layers["fg"]["z"], layers["fg"]["bgr"],
            layers["fg"]["pollen"]
        )
        fg_blur = cv2.GaussianBlur(fg_canvas, (0, 0), sigmaX=2.4)
        mask_diff = np.any(fg_canvas != canvas, axis=-1)
        mask_float = cv2.GaussianBlur(mask_diff.astype(np.float32), (0, 0), sigmaX=2.0)[..., None]
        canvas = np.clip(canvas * (1.0 - mask_float) + fg_blur * mask_float, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------------
    # 5. Overlay 3D Parametric Wireframe Splines (With Soft Additive Glow)
    # ------------------------------------------------------------------------
    if wire_lines:
        wire_layer = np.zeros_like(canvas)
        wire_lines.sort(key=lambda item: item[1], reverse=True)
        for pts_sub, _, wc_bgr, thick in wire_lines:
            cv2.polylines(
                wire_layer, [pts_sub], isClosed=False, color=wc_bgr,
                thickness=thick, lineType=cv2.LINE_AA, shift=SHIFT
            )
        canvas = cv2.addWeighted(canvas, 1.0, wire_layer, 0.42, 0.0)

    # ------------------------------------------------------------------------
    # 6. Atmospheric Bokeh Dust & Growth Wavefront Sparkles
    # ------------------------------------------------------------------------
    particle_layer = np.zeros_like(canvas)
    rng_p = np.random.RandomState(108)
    n_ambient = 58
    amb_pos = rng_p.uniform([-1.1, -1.7, -0.9], [1.1, 1.8, 0.8], size=(n_ambient, 3)).astype(np.float32)
    amb_phase = rng_p.uniform(0.0, 2.0 * math.pi, size=n_ambient)
    amb_pos[:, 1] += 0.14 * t + 0.07 * np.sin(t * 1.3 + amb_phase)
    amb_pos[:, 1] = ((amb_pos[:, 1] + 1.7) % 3.5) - 1.7
    amb_pos[:, 0] += 0.09 * np.cos(t * 1.0 + amb_phase)

    asx, asy, azd, _ = project_points(amb_pos, cam_rot, cam_target, cam_dist, fov_scale)
    for i in range(n_ambient):
        px, py = int(round(asx[i])), int(round(asy[i]))
        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
            twinkle = (0.35 + 0.65 * math.sin(t * 2.4 + amb_phase[i])) * bg_fade
            rad = int(round((1.5 + 2.8 * (i % 3)) * (4.2 / max(2.2, azd[i]))))
            col = (
                int(145 * twinkle),
                int(180 * twinkle),
                int(245 * twinkle)
            )
            cv2.circle(particle_layer, (px, py), max(1, rad), col, -1, lineType=cv2.LINE_AA)

    for px, py, bgr, rad in wave_particles[:240]:
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < WIDTH and 0 <= iy < HEIGHT:
            cv2.circle(particle_layer, (ix, iy), max(1, int(round(rad))), bgr, -1, lineType=cv2.LINE_AA)

    particle_blur = cv2.GaussianBlur(particle_layer, (0, 0), sigmaX=3.5)
    canvas = cv2.addWeighted(canvas, 1.0, particle_blur, 0.80, 0.0)
    canvas = cv2.addWeighted(canvas, 1.0, particle_layer, 0.45, 0.0)

    # ------------------------------------------------------------------------
    # 7. Subtle Optical Highlight Bloom & Museum Botanical Telemetry HUD
    # ------------------------------------------------------------------------
    bright = np.maximum(canvas.astype(np.float32) - 210.0, 0.0).astype(np.uint8)
    bloom = cv2.GaussianBlur(bright, (0, 0), sigmaX=14.0)
    canvas = cv2.addWeighted(canvas, 1.0, bloom, 0.30, 0.0)

    if SHOW_HUD:
        draw_botanical_hud(canvas, t)

    return canvas


def draw_botanical_hud(canvas, t):
    """Draw clean, understated botanical specimen telemetry in the 9:16 safe zones."""
    hud_alpha = smootherstep(0.3, 1.2, t) * (1.0 - 0.40 * smootherstep(15.0, 16.8, t))
    if hud_alpha <= 0.02:
        return

    overlay = canvas.copy()
    c_main = (220, 225, 240)
    c_accent = (195, 185, 242)
    c_muted = (145, 148, 168)

    cv2.putText(
        overlay, "PRUNUS SERRULATA  //  SAKURA", (68, 118),
        cv2.FONT_HERSHEY_SIMPLEX, 0.74, c_main, 2, cv2.LINE_AA
    )
    cv2.putText(
        overlay, "3D PROCEDURAL BOTANICAL SPECIMEN  //  CORYMB INFLORESCENCE", (68, 152),
        cv2.FONT_HERSHEY_SIMPLEX, 0.41, c_muted, 1, cv2.LINE_AA
    )
    cv2.line(overlay, (68, 172), (420, 172), (95, 100, 125), 1, cv2.LINE_AA)

    if t < 3.4:
        stage_str = "STAGE 01 // LIGNEOUS BRANCHING & LENTICEL BARK"
        eq_str = "B(u,v) : Cubic Bezier Parallel Transport + Nodal Swelling"
    elif t < 6.2:
        stage_str = "STAGE 02 // CORYMB PEDICELS, HYPANTHIA & BUD SWELLING"
        eq_str = "H(u) : Urceolate Calyx Cup + Imbricate Sepal Scales"
    elif t < 10.0:
        stage_str = "STAGE 03 // EMARGINATE 5-PETAL UNFOLDING & SSS SHADING"
        eq_str = "P(u,v) : Bilobed Apical Cleft + Concave Cupping Unfurl"
    elif t < 13.6:
        stage_str = "STAGE 04 // FULL BLOOM, 30 STAMENS & SERRULATE FOLIAGE"
        eq_str = "Androecium Whorls + Golden Anthers + Pinnate Leaf Blades"
    else:
        stage_str = "STAGE 05 // CINEMATIC SAKURA SPECIMEN & DRIFTING PETALS"
        eq_str = "3D Depth-of-Field Showcase + Tumbling Petal Aerodynamics"

    y_bot = HEIGHT - 135
    cv2.putText(
        overlay, stage_str, (68, y_bot),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46, c_accent, 1, cv2.LINE_AA
    )
    cv2.putText(
        overlay, eq_str, (68, y_bot + 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.39, c_muted, 1, cv2.LINE_AA
    )

    prog = min(1.0, t / DURATION)
    bar_w = WIDTH - 136
    cv2.rectangle(overlay, (68, y_bot + 46), (68 + bar_w, y_bot + 50), (55, 58, 72), -1)
    cv2.rectangle(
        overlay, (68, y_bot + 46), (68 + int(bar_w * prog), y_bot + 50),
        (190, 178, 245), -1
    )

    cv2.addWeighted(overlay, hud_alpha * 0.82, canvas, 1.0 - hud_alpha * 0.82, 0.0, dst=canvas)


# ============================================================================
# CLI ENTRY POINT (PREVIEW KEYFRAMES OR FULL MP4 RENDER)
# ============================================================================
def render_preview_keyframes(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    init_worker(WIDTH, HEIGHT, SHOW_HUD)
    test_times = [2.2, 5.2, 8.8, 12.6, 16.2]
    aspect_tag = f"{WIDTH}x{HEIGHT}"
    for sec in test_times:
        f_idx = int(round((sec / DURATION) * (TOTAL_FRAMES - 1)))
        t0 = time.time()
        img = render_frame(f_idx, TOTAL_FRAMES)
        out_path = os.path.join(out_dir, f"preview_{aspect_tag}_{sec:.1f}s.png")
        cv2.imwrite(out_path, img)
        print(f"Rendered {out_path} in {time.time() - t0:.2f}s")


def find_ffmpeg():
    """Locate FFmpeg binary: project folder -> imageio_ffmpeg -> system PATH."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "ffmpeg.exe"),
        os.path.join(here, "ffmpeg"),
        os.path.join(here, "bin", "ffmpeg.exe"),
        os.path.join(here, "bin", "ffmpeg"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass

    return shutil.which("ffmpeg")


def render_full_video(output_mp4, max_frames=None):
    target_frames = min(TOTAL_FRAMES, max_frames) if max_frames else TOTAL_FRAMES
    print(f"Rendering {target_frames}/{TOTAL_FRAMES} frames at {WIDTH}x{HEIGHT} ({FPS} fps, {DURATION:.1f}s)...")
    t_start = time.time()

    ffmpeg_bin = find_ffmpeg()
    proc = None
    cv_writer = None

    if ffmpeg_bin:
        print(f"Using FFmpeg binary: {ffmpeg_bin}")
        ffmpeg_cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{WIDTH}x{HEIGHT}",
            "-pix_fmt", "bgr24",
            "-r", str(FPS),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "17",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_mp4,
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        print("Notice: FFmpeg not detected. Using OpenCV VideoWriter (mp4v codec) fallback...")
        cv_writer = cv2.VideoWriter(
            output_mp4,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (WIDTH, HEIGHT)
        )
        if not cv_writer.isOpened():
            raise RuntimeError(f"Failed to initialize OpenCV VideoWriter for {output_mp4}")

    num_workers = min(8, os.cpu_count() or 4)
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(WIDTH, HEIGHT, SHOW_HUD)) as pool:
        for idx, frame in enumerate(pool.imap(render_frame, range(target_frames), chunksize=4)):
            if proc is not None:
                proc.stdin.write(frame.tobytes())
            elif cv_writer is not None:
                cv_writer.write(frame)

            if (idx + 1) % 30 == 0 or (idx + 1) == target_frames:
                elapsed = time.time() - t_start
                fps_actual = (idx + 1) / max(1e-5, elapsed)
                print(
                    f"  [{idx + 1:3d}/{target_frames}] "
                    f"({100.0 * (idx + 1) / target_frames:5.1f}%) - {fps_actual:.1f} fps"
                )

    if proc is not None:
        proc.stdin.close()
        stderr_out = proc.stderr.read().decode("utf-8", errors="ignore")
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {ret}:\n{stderr_out}")
    elif cv_writer is not None:
        cv_writer.release()

    print(f"Done! Saved high-quality MP4 to: {output_mp4} in {time.time() - t_start:.1f}s")


def play_video(video_path):
    """Interactive loop playback of the generated animation in an OpenCV window."""
    if not os.path.isfile(video_path):
        print(f"Error: '{video_path}' not found. Please render it first.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open '{video_path}'.")
        return

    win_title = "Sakura 3D Animation [Space: Pause | Q/Esc: Quit]"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 960, 540 if WIDTH > HEIGHT else 960)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = max(1, int(round(1000.0 / fps)))
    paused = False

    print(f"Playing '{video_path}' (Press SPACE to pause, Q or ESC to quit)...")
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cv2.imshow(win_title, frame)

        key = cv2.waitKey(delay if not paused else 30) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key == 32:  # Space
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    out_folder = os.path.dirname(os.path.abspath(__file__))

    # CLI Options
    aspect = "portrait"
    for i, arg in enumerate(sys.argv):
        if arg == "--aspect" and i + 1 < len(sys.argv):
            aspect = sys.argv[i + 1].lower()
        elif arg == "--landscape":
            aspect = "landscape"
        elif arg == "--portrait":
            aspect = "portrait"

    if "--hud" in sys.argv:
        SHOW_HUD = True

    if aspect == "landscape":
        WIDTH, HEIGHT = 1920, 1080
        video_filename = "sakura_landscape.mp4"
    else:
        WIDTH, HEIGHT = 1080, 1920
        video_filename = "sakura_portrait.mp4"

    STUDIO_BG = create_sakura_studio_backdrop()
    video_path = os.path.join(out_folder, video_filename)

    if "--play" in sys.argv or "--view" in sys.argv:
        play_video(video_path)
    elif "--preview" in sys.argv:
        render_preview_keyframes(out_folder)
    else:
        frames_arg = None
        for i, arg in enumerate(sys.argv):
            if arg == "--frames" and i + 1 < len(sys.argv):
                try:
                    frames_arg = int(sys.argv[i + 1])
                except ValueError:
                    pass
        render_full_video(video_path, max_frames=frames_arg)
