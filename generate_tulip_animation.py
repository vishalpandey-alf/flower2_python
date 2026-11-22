#!/usr/bin/env python3
"""
Cinematic 3D Blue Tulip (Tulipa, Stylized Sapphire-Blue Cultivar) Botanical Animation
=====================================================================================
Generates an authentic 3D botanical animation of a bouquet of Blue Tulips
growing and blooming step by step:
  - Stage 1 (0.0s - 3.0s): Slender, waxy curved green stems & lanceolate leaves growing
                           upward with glaucous sheen and natural Bishop frame transport.
  - Stage 2 (3.0s - 6.0s): Closed, elegant teardrop-shaped buds forming at stem tips,
                           tinted pale cyan-green at the apex.
  - Stage 3 (6.0s - 9.0s): 6 tepals (3 outer, 3 inner) separate and open outward from
                           the goblet cup with soft curl, longitudinal venation & SSS shading.
  - Stage 4 (9.0s - 13.0s): 6 golden stamens with dehiscing amber anthers & central 3-lobed
                            pistil become visible and mature.
  - Stage 5 (13.0s - 18.0s): Full radiant open bloom, parametric orbital camera rotation,
                            ice-blue stardust & golden pollen sparkles.
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
# VIDEO & RENDER CONFIGURATION
# ============================================================================
WIDTH = 1920
HEIGHT = 1080
FPS = 30
DURATION = 18.0
TOTAL_FRAMES = int(FPS * DURATION)  # 540 frames
SHIFT = 3
SUBPIX = 1 << SHIFT
SHOW_HUD = False


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
                 wire_color=(0.35, 0.65, 0.95), wire_stride_u=4, wire_stride_v=3,
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

        if self.anchor is not None and self.scale_t0 is not None:
            sc_prog = smootherstep(self.scale_t0, self.scale_t1, t)
            sc = self.scale_init + (1.0 - self.scale_init) * sc_prog
            if sc < 0.999:
                P = self.anchor + (P - self.anchor) * sc

        if wind_vec is not None and self.anchor is not None and self.kind in ("petal", "leaf", "stamen", "stem"):
            dist_from_anchor = np.linalg.norm(P - self.anchor, axis=-1, keepdims=True)
            sway_factor = (dist_from_anchor ** 1.35) * (0.26 if self.kind == "petal" else 0.20)
            P = P + sway_factor * wind_vec

        return P, C, b_prog


def slice_mesh_along_u(P, C, prog):
    """
    Smoothly slice a parametric mesh (Nu, Nv, 3) along longitudinal u-axis.
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
# PROCEDURAL 3D BOTANICAL GENERATORS (BLUE TULIP / TULIPA)
# ============================================================================
def generate_tulip_stem(p0, p1, p2, p3, r_base, r_tip, Nu, Nv,
                        wire_t0, wire_t1, surf_t0, surf_t1,
                        rng_seed=42, dof_layer="mid"):
    """
    Generate a slender, waxy, curved green tulip stem along a cubic Bezier spline
    with parallel Bishop transport frame.
    """
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    pos, T = eval_cubic_bezier(p0, p1, p2, p3, u)
    T = normalize_v3(T)
    N1, N2 = compute_bishop_frame(T)

    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    theta = (v + 1.0) * math.pi

    P = np.zeros((Nu, Nv, 3), dtype=np.float32)
    C = np.zeros((Nu, Nv, 3), dtype=np.float32)

    # Stem color: smooth waxy green (#3F6B3A -> #6E9A4A) with subtle longitudinal sheen
    c_stem_base = np.array([0.25, 0.42, 0.23], dtype=np.float32)
    c_stem_mid = np.array([0.31, 0.50, 0.26], dtype=np.float32)
    c_stem_top = np.array([0.43, 0.60, 0.29], dtype=np.float32)

    for i in range(Nu):
        ui = u[i]
        r = r_base * (1.0 - ui) + r_tip * ui
        P[i] = pos[i] + r * (np.cos(theta)[:, None] * N1[i] + np.sin(theta)[:, None] * N2[i])

        if ui < 0.5:
            mix = ui / 0.5
            col = (1.0 - mix) * c_stem_base + mix * c_stem_mid
        else:
            mix = (ui - 0.5) / 0.5
            col = (1.0 - mix) * c_stem_mid + mix * c_stem_top
        sheen = 0.96 + 0.04 * np.cos(4.0 * theta)[:, None]
        C[i] = col * sheen

    el = BotanicalElement(
        kind="stem", P=P, C=C,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.22, spec_strength=0.28, shininess=28.0,
        wire_color=(0.42, 0.78, 0.45), wire_stride_u=5, wire_stride_v=4,
        dof_layer=dof_layer
    )
    return el, pos, T


def generate_tulip_bud(tip_pos, tip_dir, scale, wire_t0, wire_t1, surf_t0, surf_t1,
                       rng_seed=10, dof_layer="mid"):
    """
    Generate an authentic closed teardrop-shaped tulip bud in Stage 2,
    tinted pale cyan-green at the apex, hugging the top of the stem.
    """
    basis = build_basis_from_dir(tip_dir)
    Nu, Nv = 36, 25
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')
    theta = (V + 1.0) * math.pi

    length = 0.28 * scale
    r_max = 0.088 * scale

    env = np.maximum(0.0, np.sin(math.pi * (U ** 0.58))) ** 0.85
    imbricate = 1.0 + 0.08 * np.cos(3.0 * theta + 0.8 * U)
    r_prof = r_max * env * imbricate

    y_loc = length * U
    x_loc = r_prof * np.cos(theta)
    z_loc = r_prof * np.sin(theta)
    P_local = np.stack([x_loc, y_loc, z_loc], axis=-1)
    P_world = tip_pos + P_local @ basis.T

    # Bud color palette: pale green base (#3B6B38) transitioning to deep sapphire (#1F4FBF)
    # and soft glaucous pale cyan-blue tip (#8FC3FF / #6EA8E8)
    c_base = np.array([0.23, 0.42, 0.22], dtype=np.float32)
    c_mid = np.array([0.12, 0.31, 0.75], dtype=np.float32)   # #1F4FBF
    c_tip = np.array([0.48, 0.72, 0.95], dtype=np.float32)   # pale cyan-blue

    U_c = U[..., None]
    col = np.where(
        U_c < 0.35,
        (1.0 - (U_c / 0.35)) * c_base + (U_c / 0.35) * c_mid,
        (1.0 - ((U_c - 0.35) / 0.65)) * c_mid + ((U_c - 0.35) / 0.65) * c_tip
    )
    striation = (0.95 + 0.05 * np.cos(12.0 * theta + U * 2.0))[..., None]
    C = col * striation

    return BotanicalElement(
        kind="bud", P=P_world, C=C,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.32, spec_strength=0.25, shininess=24.0,
        wire_color=(0.35, 0.68, 0.95), wire_stride_u=6, wire_stride_v=4,
        anchor=tip_pos, scale_t0=surf_t0, scale_t1=surf_t1 + 0.8, scale_init=0.25,
        dof_layer=dof_layer
    )


def generate_tulip_tepal(center, basis, azimuth, scale, is_inner, max_openness,
                         wire_t0, wire_t1, surf_t0, surf_t1, bloom_t0, bloom_t1,
                         rng_seed=100, dof_layer="mid"):
    """
    Generate a single 3D Tulip tepal:
      - Botanically authentic: 6 tepals total (3 outer whorl, 3 inner whorl)
      - Outer tepals slightly narrower with soft ogee apex; inner tepals slightly broader & cupped
      - Hero sapphire-blue palette: base #1F4FBF, mid #3B7BE8, edges #8FC3FF, throat #0B2A7A
      - Full morphological unfolding: forms closed teardrop bud in Stage 2 (P_closed, C_closed),
        unfolds outward from goblet cup in Stage 3 (P_open, C_open) with soft SSS and curl
    """
    rng = np.random.RandomState(rng_seed)
    Nu, Nv = 44, 31
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')

    len_var = rng.uniform(0.96, 1.04)
    wid_var = rng.uniform(0.95, 1.05)
    az_jitter = rng.uniform(-0.03, 0.03)

    length = (0.44 if is_inner else 0.46) * scale * len_var
    w_max = (0.165 if is_inner else 0.145) * scale * wid_var

    # Elliptical / ogee perimeter profile (widest at 55% height, softly pointed apex)
    abs_V = np.abs(V)
    u_prof = np.maximum(0.0, np.sin(math.pi * np.clip(U ** 0.72, 0.0, 1.0))) ** 0.88
    # Gentle taper to tip with softly pointed/ogee apex
    tip_taper = (np.maximum(0.0, 1.0 - U) ** 0.45) * smoothstep(0.70, 1.0, U) + (1.0 - smoothstep(0.70, 1.0, U))
    w_edge = w_max * u_prof * tip_taper * (1.0 - 0.15 * (abs_V ** 2))

    def build_tepal_state(open_amt):
        if open_amt < 0.04:
            # Closed teardrop bud state (Stage 2)
            bud_len = 0.32 * scale * len_var
            bud_r_max = (0.078 if is_inner else 0.086) * scale * wid_var
            env_u = np.maximum(0.0, np.sin(math.pi * (U ** 0.60))) ** 0.85
            # Imbricate overlapping in 3-outer / 3-inner configuration
            imb_r = bud_r_max * env_u * (1.0 + (0.04 if is_inner else -0.04) * V)
            wrap_ang = azimuth + az_jitter + V * 0.58 + (0.08 if is_inner else -0.08) * U
            px = imb_r * np.cos(wrap_ang)
            pz = imb_r * np.sin(wrap_ang)
            py = bud_len * U + 0.02 * scale * (1.0 - U)
            P_bud_loc = np.stack([px, py, pz], axis=-1)
            return center + P_bud_loc @ basis.T

        # Open / blooming goblet state (Stage 3 -> Stage 5)
        # Flare angle increases with open_amt, outer tepals flare slightly more than inner
        flare_mult = 1.08 if not is_inner else 0.94
        theta_base = 0.18 + 0.22 * open_amt
        theta_mid = 0.32 + 0.68 * open_amt * flare_mult
        theta_tip = 0.20 + 0.88 * open_amt * flare_mult

        Ang = (
            theta_base * (1.0 - smoothstep(0.0, 0.45, U))
            + theta_mid * smoothstep(0.0, 0.45, U) * (1.0 - smoothstep(0.40, 1.0, U))
            + theta_tip * smoothstep(0.40, 1.0, U)
        )

        r_mid = (0.024 if is_inner else 0.030) * scale + (length * U) * np.sin(Ang * 0.84)
        y_mid = 0.006 * scale + (length * U) * np.cos(Ang * 0.84)

        n_r = np.cos(Ang)
        n_y = -np.sin(Ang)

        # Delicate concave goblet bowl cupping + central midrib ridge
        cup = -0.038 * scale * (1.10 - 0.30 * open_amt) * (V ** 2) * np.sin(math.pi * U)
        midrib = 0.007 * scale * np.exp(-((V / 0.18) ** 2)) * (1.0 - 0.35 * U)
        ruffle = 0.004 * scale * open_amt * (abs_V ** 2.0) * smoothstep(0.55, 1.0, U) * np.sin(V * 3.0 * math.pi)

        disp_n = cup + midrib + ruffle
        x_flat = w_edge * V

        R_final = r_mid + disp_n * n_r
        Y_final = y_mid + disp_n * n_y

        az = azimuth + az_jitter
        ca, sa = math.cos(az), math.sin(az)
        px = R_final * ca - x_flat * sa
        pz = R_final * sa + x_flat * ca
        py = Y_final

        P_loc = np.stack([px, py, pz], axis=-1)
        return center + P_loc @ basis.T

    P_closed = build_tepal_state(0.0)
    P_open = build_tepal_state(max_openness)

    # Hero Blue Tulip Color Palette (Linear RGB from sRGB)
    # Throat: #0B2A7A (0.043, 0.165, 0.478)
    # Base:   #1F4FBF (0.122, 0.310, 0.749)
    # Mid:    #3B7BE8 (0.231, 0.482, 0.910)
    # Edge:   #8FC3FF (0.561, 0.765, 1.000)
    # Subtle violet accent #5B5BE0 (0.357, 0.357, 0.878)
    c_throat = np.array([0.04, 0.16, 0.48], dtype=np.float32)
    c_base = np.array([0.12, 0.31, 0.75], dtype=np.float32)
    c_mid = np.array([0.23, 0.48, 0.91], dtype=np.float32)
    c_edge = np.array([0.56, 0.76, 1.00], dtype=np.float32)
    c_violet = np.array([0.36, 0.36, 0.88], dtype=np.float32)
    c_vein = np.array([0.16, 0.38, 0.84], dtype=np.float32)

    throat_mask = np.exp(-((U / 0.22) ** 2)) * np.exp(-((V / 0.50) ** 2))
    u_grad = smoothstep(0.10, 0.85, U)[..., None]
    edge_mask = (abs_V ** 1.8)[..., None] * smoothstep(0.40, 1.0, U)[..., None]

    C_open = (1.0 - u_grad) * c_base + u_grad * c_mid
    C_open = (1.0 - throat_mask[..., None]) * C_open + throat_mask[..., None] * c_throat
    C_open = (1.0 - edge_mask) * C_open + edge_mask * c_edge
    # 10% subtle violet shift along lateral flanks
    flank_mask = smoothstep(0.35, 0.85, abs_V)[..., None] * 0.10
    C_open = (1.0 - flank_mask) * C_open + flank_mask * c_violet

    # Fine longitudinal venation
    vein_coord = V * 11.0 + U * 0.5
    vein_pattern = ((0.5 + 0.5 * np.cos(vein_coord * math.pi)) ** 6)[..., None]
    vein_weight = 0.18 * (1.0 - 0.40 * U[..., None])
    C_open = (1.0 - vein_weight * vein_pattern) * C_open + (vein_weight * vein_pattern) * c_vein

    # Closed bud colors (Stage 2): tinted pale cyan-green at tip and deep sapphire at base
    c_bud_base = np.array([0.10, 0.28, 0.68], dtype=np.float32)
    c_bud_tip = np.array([0.38, 0.68, 0.88], dtype=np.float32)
    C_closed = (1.0 - U[..., None]) * c_bud_base + U[..., None] * c_bud_tip

    return BotanicalElement(
        kind="petal", P=P_open, P_closed=P_closed, C=C_open, C_closed=C_closed,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        bloom_t0=bloom_t0, bloom_t1=bloom_t1,
        extra_wire_t0=bloom_t0 - 0.2, extra_wire_t1=bloom_t1,
        sss_strength=0.38, spec_strength=0.22, shininess=24.0,
        wire_color=(0.35, 0.68, 0.98), wire_stride_u=6, wire_stride_v=4,
        anchor=center, scale_t0=surf_t0, scale_t1=surf_t1 + 0.5, scale_init=0.35,
        dof_layer=dof_layer
    )


def generate_tulip_androecium_and_pistil(center, basis, scale, wire_t0, wire_t1, surf_t0, surf_t1,
                                         bloom_t0, bloom_t1, rng_seed=500, dof_layer="mid"):
    """
    Generate the internal reproductive anatomy of a tulip:
      - 1 central pale green-white pistil with prominent 3-lobed stigma (#C8E6C9 -> #E8F5E9)
      - 6 upright stamens with golden-amber anthers (#E8B923) crowned with pollen
    """
    rng = np.random.RandomState(rng_seed)
    elements = []

    Nu, Nv = 16, 8
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    theta = (v + 1.0) * math.pi

    # 1. Central 3-lobed Pistil (Ovary + 3-lobed sessile stigma)
    p_len = 0.165 * scale
    p_pos = np.stack([np.zeros_like(u), p_len * u, np.zeros_like(u)], axis=-1)
    P_pistil = np.zeros((Nu, Nv, 3), dtype=np.float32)
    C_pistil = np.zeros((Nu, Nv, 3), dtype=np.float32)

    c_ovary_base = np.array([0.45, 0.68, 0.38], dtype=np.float32)
    c_stigma = np.array([0.82, 0.92, 0.78], dtype=np.float32)

    for i in range(Nu):
        ui = u[i]
        if ui < 0.80:
            # Triangular prismatic ovary
            r = scale * (0.014 + 0.004 * ui) * (1.0 + 0.15 * math.cos(3.0 * theta[0]))
        else:
            # 3-lobed recurved stigma crest
            st_u = (ui - 0.80) / 0.20
            r = scale * (0.018 + 0.024 * math.sin(st_u * math.pi))
        P_pistil[i] = p_pos[i] + np.stack([r * np.cos(theta), np.zeros_like(theta), r * np.sin(theta)], axis=-1)
        C_pistil[i] = (1.0 - ui) * c_ovary_base + ui * c_stigma

    P_pistil_world = center + P_pistil @ basis.T
    elements.append(BotanicalElement(
        kind="stamen", P=P_pistil_world, C=C_pistil,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.25, spec_strength=0.28, shininess=22.0,
        wire_color=(0.65, 0.88, 0.60), wire_stride_u=5, wire_stride_v=3,
        anchor=center, dof_layer=dof_layer
    ))

    # 2. 6 Golden Stamens (#E8B923 anthers + dark navy/green filament bases)
    c_fil_base = np.array([0.15, 0.30, 0.45], dtype=np.float32)
    c_fil_top = np.array([0.65, 0.75, 0.40], dtype=np.float32)
    c_anther_core = (35, 185, 232)  # BGR for golden #E8B923
    c_anther_rim = (20, 145, 195)   # BGR rim

    stamen_count = 6
    s_len = 0.145 * scale
    r0 = 0.024 * scale

    for k in range(stamen_count):
        az = (k / float(stamen_count)) * 2.0 * math.pi + rng.uniform(-0.04, 0.04)
        flare_open = 0.32 + rng.uniform(0.0, 0.08)
        ca, sa = math.cos(az), math.sin(az)
        rad_dir = np.array([ca, 0.0, sa], dtype=np.float32)
        up_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        def make_filament(flare_a):
            q0 = r0 * rad_dir + 0.005 * scale * up_dir
            q1 = (r0 + 0.22 * s_len * math.sin(flare_a * 0.7)) * rad_dir + (0.35 * s_len * math.cos(flare_a * 0.7)) * up_dir
            q2 = (r0 + 0.58 * s_len * math.sin(flare_a)) * rad_dir + (0.72 * s_len * math.cos(flare_a)) * up_dir
            q3 = (r0 + 0.88 * s_len * math.sin(flare_a * 0.95)) * rad_dir + (1.02 * s_len * math.cos(flare_a * 0.90)) * up_dir
            pos_c, _ = eval_cubic_bezier(q0, q1, q2, q3, u)
            P_f = np.zeros((Nu, Nv, 3), dtype=np.float32)
            r_fil = 0.0035 * scale
            for i in range(Nu):
                rf = r_fil * (1.0 - 0.25 * u[i])
                P_f[i] = pos_c[i] + np.stack([rf * np.cos(theta), np.zeros_like(theta), rf * np.sin(theta)], axis=-1)
            return P_f, pos_c[-1]

        P_f_open, anther_open_loc = make_filament(flare_open)
        P_f_closed, anther_closed_loc = make_filament(0.08)

        P_f_open_w = center + P_f_open @ basis.T
        P_f_closed_w = center + P_f_closed @ basis.T

        C_f = np.zeros((Nu, Nv, 3), dtype=np.float32)
        for i in range(Nu):
            C_f[i] = (1.0 - u[i]) * c_fil_base + u[i] * c_fil_top

        anther_open_w = center + anther_open_loc @ basis.T
        anther_closed_w = center + anther_closed_loc @ basis.T

        pollen_tip = (
            anther_closed_w, anther_open_w,
            max(2, int(round(4.0 * scale))),
            c_anther_core, c_anther_rim
        )

        elements.append(BotanicalElement(
            kind="stamen", P=P_f_open_w, P_closed=P_f_closed_w, C=C_f,
            wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
            bloom_t0=bloom_t0, bloom_t1=bloom_t1,
            sss_strength=0.28, spec_strength=0.30, shininess=22.0,
            wire_color=(0.95, 0.78, 0.25), wire_stride_u=5, wire_stride_v=3,
            anchor=center, dof_layer=dof_layer,
            pollen_tip=pollen_tip
        ))

    return elements


def generate_tulip_leaf(origin, direction, length, width, azimuth, tilt,
                        wire_t0, wire_t1, surf_t0, surf_t1,
                        rng_seed=200, dof_layer="mid"):
    """
    Generate an authentic lanceolate tulip leaf:
      - Long, gently arched, blade clasps stem base and arches outward
      - Distinct central longitudinal fold/crease
      - Glaucous blue-green waxy color with powder bloom
    """
    Nu, Nv = 36, 19
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')

    # Lanceolate taper: widest at ~35% length, tapering gracefully to an acute point
    w_prof = width * np.sin(math.pi * (U ** 0.65)) * (1.0 - 0.70 * U)

    def build_leaf(unfold_t):
        reach = length * (0.35 + 0.65 * unfold_t) * U
        # Graceful outward arch
        elev = length * (0.28 * U - 0.42 * (U ** 2.1) * (0.5 + 0.5 * unfold_t))
        # Central V-shaped fold relaxing as leaf expands
        fold_angle = 0.85 * (1.0 - unfold_t) + 0.32 * unfold_t
        v_fold = fold_angle * width * np.abs(V) * (1.0 - 0.30 * U)

        x_loc = w_prof * V
        y_loc = reach
        z_loc = elev + v_fold

        P_loc = np.stack([x_loc, y_loc, z_loc], axis=-1)
        rot = rot_y(azimuth) @ rot_x(tilt)
        return origin + P_loc @ rot.T

    P_closed = build_leaf(0.20)
    P_open = build_leaf(1.00)

    # Glaucous blue-green palette (#2D5A46 base, #488266 mid, #6EA890 glaucous bloom)
    c_deep = np.array([0.18, 0.35, 0.28], dtype=np.float32)
    c_mid = np.array([0.28, 0.51, 0.40], dtype=np.float32)
    c_bloom = np.array([0.43, 0.66, 0.56], dtype=np.float32)
    c_midrib = np.array([0.50, 0.74, 0.60], dtype=np.float32)

    midrib_mask = np.exp(-((V / 0.10) ** 2))[..., None]
    glaucous_f = (0.55 * (U ** 1.4)[..., None] + 0.45 * (np.abs(V) ** 1.8)[..., None])

    C = (1.0 - glaucous_f) * c_mid + glaucous_f * c_bloom
    C = (1.0 - 0.45 * midrib_mask) * C + 0.45 * midrib_mask * c_midrib

    return BotanicalElement(
        kind="leaf", P=P_open, P_closed=P_closed, C=C,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        bloom_t0=surf_t0, bloom_t1=surf_t1 + 1.2,
        sss_strength=0.26, spec_strength=0.28, shininess=26.0,
        wire_color=(0.42, 0.76, 0.60), wire_stride_u=6, wire_stride_v=4,
        anchor=origin, scale_t0=surf_t0, scale_t1=surf_t1 + 0.9, scale_init=0.30,
        dof_layer=dof_layer
    )


# ============================================================================
# DRIFTING SAPPHIRE TEPAL TEMPLATE (STAGE 5 REVEAL)
# ============================================================================
def build_drifting_tepals_template():
    Nu, Nv = 24, 17
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')

    length = 0.38
    w_max = 0.14
    w_edge = w_max * np.sin(math.pi * (U ** 0.75)) * (1.0 - 0.2 * (V ** 2))
    cup = -0.024 * (V ** 2) * np.sin(math.pi * U)

    P = np.stack([w_edge * V, length * U, cup], axis=-1)

    c_base = np.array([0.12, 0.31, 0.75], dtype=np.float32)
    c_mid = np.array([0.23, 0.48, 0.91], dtype=np.float32)
    c_edge = np.array([0.56, 0.76, 1.00], dtype=np.float32)

    C = (1.0 - U[..., None]) * c_base + U[..., None] * c_mid
    C = (1.0 - (np.abs(V) ** 2)[..., None] * 0.4) * C + (np.abs(V) ** 2)[..., None] * 0.4 * c_edge

    rng = np.random.RandomState(88)
    specs = []
    for i in range(24):
        specs.append({
            "start_pos": np.array([
                rng.uniform(-0.65, 0.75),
                rng.uniform(-0.85, 1.35),
                rng.uniform(-0.45, 0.55)
            ], dtype=np.float32),
            "vel": np.array([
                rng.uniform(-0.06, 0.08),
                rng.uniform(-0.16, -0.07),
                rng.uniform(-0.05, 0.06)
            ], dtype=np.float32),
            "scale": rng.uniform(0.72, 1.15),
            "rot_rates": np.array([rng.uniform(0.6, 1.6), rng.uniform(0.8, 2.0), rng.uniform(0.5, 1.4)]),
            "rot_phases": np.array([rng.uniform(0, 6.28), rng.uniform(0, 6.28), rng.uniform(0, 6.28)]),
            "t_spawn": 13.2 + rng.uniform(0.0, 2.8),
            "dof": "fg" if (i % 6 == 0) else ("bg" if (i % 5 == 0) else "mid")
        })

    return P, C, specs


DRIFT_P_BASE, DRIFT_C_BASE, DRIFT_SPECS = build_drifting_tepals_template()


# ============================================================================
# COMPLETE BLUE TULIP SCENE ASSEMBLY (7 FLOWERS ACROSS 5 STAGES)
# ============================================================================
def build_tulip_scene():
    """
    Construct the complete 3D Blue Tulip bouquet across 5 stages:
      - 7 stems in one cohesive diagonal bouquet composition rising from lower-left
      - 14 glaucous lanceolate leaves clasping stem bases
      - 7 flowers:
          * 2 tight/swelling buds in Stage 2 that stay buds
          * 5 main flowers that swell in Stage 2 and bloom in Stage 3 -> 5
      - 6 tepals (3 outer, 3 inner) per flower + 6 golden stamens + 3-lobed pistil
      - Stage mapping exact to sakura timing:
          * Stage 1: 0.0s - 3.0s (stems & leaves)
          * Stage 2: 3.0s - 6.0s (buds)
          * Stage 3: 6.0s - 9.0s (tepal unfolding)
          * Stage 4: 9.0s - 13.0s (stamen & anther maturation)
          * Stage 5: 13.0s - 18.0s (radiant anthesis + drifting sapphire tepals)
    """
    elements = []

    # Bouquet base cluster origin
    base_origin = np.array([-0.45, -2.15, -0.05], dtype=np.float32)

    # 7 Stems defining the diagonal silhouette (identical composition/scale to reference)
    stems_spec = [
        # 1. Primary Tall Center Stem (Hero Center Flower)
        {
            "p0": base_origin + np.array([0.02, 0.0, 0.0]),
            "p1": base_origin + np.array([0.12, 1.15, 0.10]),
            "p2": np.array([-0.05, -0.15, 0.16], dtype=np.float32),
            "p3": np.array([0.15, 0.72, 0.18], dtype=np.float32),
            "r_base": 0.042, "r_tip": 0.022,
            "wire_t0": 0.0, "wire_t1": 1.8, "surf_t0": 0.35, "surf_t1": 2.5,
            "dof": "mid", "seed": 11
        },
        # 2. Main Hero Flower Stem (Mid-Left Lower-Center)
        {
            "p0": base_origin + np.array([-0.04, 0.0, 0.04]),
            "p1": base_origin + np.array([-0.18, 0.95, 0.14]),
            "p2": np.array([-0.36, -0.55, 0.22], dtype=np.float32),
            "p3": np.array([-0.26, 0.12, 0.26], dtype=np.float32),
            "r_base": 0.040, "r_tip": 0.020,
            "wire_t0": 0.3, "wire_t1": 2.0, "surf_t0": 0.65, "surf_t1": 2.7,
            "dof": "mid", "seed": 22
        },
        # 3. Hero Mid-Right Stem
        {
            "p0": base_origin + np.array([0.06, 0.0, -0.02]),
            "p1": base_origin + np.array([0.28, 0.88, 0.12]),
            "p2": np.array([0.38, -0.28, 0.20], dtype=np.float32),
            "p3": np.array([0.52, 0.38, 0.22], dtype=np.float32),
            "r_base": 0.038, "r_tip": 0.020,
            "wire_t0": 0.5, "wire_t1": 2.2, "surf_t0": 0.85, "surf_t1": 2.9,
            "dof": "mid", "seed": 33
        },
        # 4. Lower-Front Left Accent Stem
        {
            "p0": base_origin + np.array([-0.08, 0.0, 0.06]),
            "p1": base_origin + np.array([-0.25, 0.72, 0.18]),
            "p2": np.array([-0.42, -0.85, 0.25], dtype=np.float32),
            "p3": np.array([-0.48, -0.32, 0.28], dtype=np.float32),
            "r_base": 0.036, "r_tip": 0.018,
            "wire_t0": 0.8, "wire_t1": 2.4, "surf_t0": 1.10, "surf_t1": 3.1,
            "dof": "mid", "seed": 44
        },
        # 5. Top Crown Stem (Tallest Right)
        {
            "p0": base_origin + np.array([0.08, 0.0, -0.04]),
            "p1": base_origin + np.array([0.22, 1.35, 0.08]),
            "p2": np.array([0.28, 0.45, 0.14], dtype=np.float32),
            "p3": np.array([0.36, 1.38, 0.12], dtype=np.float32),
            "r_base": 0.036, "r_tip": 0.018,
            "wire_t0": 0.9, "wire_t1": 2.6, "surf_t0": 1.25, "surf_t1": 3.3,
            "dof": "mid", "seed": 55
        },
        # 6. Upper-Left Bud Stem (Closed Accent Bud)
        {
            "p0": base_origin + np.array([-0.02, 0.0, -0.05]),
            "p1": base_origin + np.array([-0.15, 1.25, 0.05]),
            "p2": np.array([-0.22, 0.52, 0.10], dtype=np.float32),
            "p3": np.array([-0.18, 1.22, 0.08], dtype=np.float32),
            "r_base": 0.032, "r_tip": 0.016,
            "wire_t0": 1.2, "wire_t1": 2.8, "surf_t0": 1.50, "surf_t1": 3.5,
            "dof": "mid", "seed": 66
        },
        # 7. Background Depth Accent Stem
        {
            "p0": base_origin + np.array([0.0, 0.0, -0.10]),
            "p1": base_origin + np.array([0.10, 0.95, -0.18]),
            "p2": np.array([0.05, 0.15, -0.25], dtype=np.float32),
            "p3": np.array([0.12, 0.85, -0.28], dtype=np.float32),
            "r_base": 0.030, "r_tip": 0.015,
            "wire_t0": 1.4, "wire_t1": 3.0, "surf_t0": 1.70, "surf_t1": 3.6,
            "dof": "bg", "seed": 77
        },
    ]

    stem_data = []
    for s in stems_spec:
        el, pos_seq, tan_seq = generate_tulip_stem(
            p0=s["p0"], p1=s["p1"], p2=s["p2"], p3=s["p3"],
            r_base=s["r_base"], r_tip=s["r_tip"],
            Nu=60, Nv=21,
            wire_t0=s["wire_t0"], wire_t1=s["wire_t1"],
            surf_t0=s["surf_t0"], surf_t1=s["surf_t1"],
            rng_seed=s["seed"], dof_layer=s["dof"]
        )
        elements.append(el)
        stem_data.append((pos_seq, tan_seq, s))

    # Lanceolate glaucous leaves clasping stems (2 per stem = 14 leaves)
    leaves_spec = [
        # (stem_idx, pos_idx, az, tilt, len, wid, t0, dof)
        (0, 14, -0.75, 0.32, 0.95, 0.12, 0.6, "mid"),
        (0, 26,  0.85, 0.38, 0.88, 0.11, 1.1, "mid"),
        (1, 12, -1.15, 0.35, 0.92, 0.12, 0.8, "mid"),
        (1, 24,  0.65, 0.40, 0.82, 0.10, 1.3, "mid"),
        (2, 12,  1.10, 0.36, 0.86, 0.11, 0.9, "mid"),
        (2, 22, -0.60, 0.42, 0.78, 0.10, 1.4, "mid"),
        (3, 10, -1.25, 0.38, 0.80, 0.11, 1.2, "mid"),
        (3, 20,  0.70, 0.45, 0.72, 0.09, 1.6, "mid"),
        (4, 15,  0.95, 0.30, 0.85, 0.11, 1.3, "mid"),
        (4, 28, -0.80, 0.35, 0.76, 0.10, 1.8, "mid"),
        (5, 16, -0.90, 0.28, 0.78, 0.10, 1.5, "mid"),
        (5, 26,  0.75, 0.32, 0.70, 0.09, 2.0, "mid"),
        (6, 14,  0.45, 0.25, 0.72, 0.09, 1.7, "bg"),
        (6, 25, -0.55, 0.30, 0.65, 0.08, 2.2, "bg"),
    ]

    for s_idx, p_idx, az, tilt, l_len, l_wid, l_t0, l_dof in leaves_spec:
        pos_seq, tan_seq, _ = stem_data[s_idx]
        orig = pos_seq[p_idx]
        tan = tan_seq[p_idx]
        elements.append(generate_tulip_leaf(
            origin=orig, direction=tan,
            length=l_len, width=l_wid,
            azimuth=az, tilt=tilt,
            wire_t0=l_t0, wire_t1=l_t0 + 1.2,
            surf_t0=l_t0 + 0.3, surf_t1=l_t0 + 1.6,
            rng_seed=100 + s_idx * 10 + p_idx,
            dof_layer=l_dof
        ))

    # Stage 2 Accent Closed Buds (Stem 5 and Stem 6)
    bud_stems = [(5, 3.4, "mid", 1.0), (6, 3.8, "bg", 0.92)]
    for s_idx, b_t0, b_dof, b_sc in bud_stems:
        pos_seq, tan_seq, _ = stem_data[s_idx]
        tip_pos = pos_seq[-1]
        tip_dir = tan_seq[-1]
        elements.append(generate_tulip_bud(
            tip_pos=tip_pos, tip_dir=tip_dir, scale=b_sc,
            wire_t0=b_t0, wire_t1=b_t0 + 1.1,
            surf_t0=b_t0 + 0.3, surf_t1=b_t0 + 1.5,
            rng_seed=250 + s_idx, dof_layer=b_dof
        ))

    # 5 Main Blooming Flowers (Stems 0, 1, 2, 3, 4)
    # Staggered bloom timing matching reference:
    flowers_spec = [
        # Flower 1: Hero Lower-Center Blossom (Stem 1)
        {
            "stem_idx": 1, "scale": 1.16, "open": 1.00,
            "rot": 0.15, "bud_t0": 3.1, "bloom_t0": 6.2, "stamen_t0": 9.2,
            "seed": 310, "dof": "mid"
        },
        # Flower 2: Primary Central Blossom (Stem 0)
        {
            "stem_idx": 0, "scale": 1.24, "open": 1.00,
            "rot": -0.22, "bud_t0": 3.3, "bloom_t0": 6.6, "stamen_t0": 9.6,
            "seed": 320, "dof": "mid"
        },
        # Flower 3: Hero Mid-Right Blossom (Stem 2)
        {
            "stem_idx": 2, "scale": 1.12, "open": 0.96,
            "rot": 0.42, "bud_t0": 3.5, "bloom_t0": 7.0, "stamen_t0": 10.0,
            "seed": 330, "dof": "mid"
        },
        # Flower 4: Lower-Left Accent Blossom (Stem 3)
        {
            "stem_idx": 3, "scale": 1.04, "open": 0.92,
            "rot": -0.45, "bud_t0": 3.7, "bloom_t0": 7.4, "stamen_t0": 10.4,
            "seed": 340, "dof": "mid"
        },
        # Flower 5: Top Crown Blossom (Stem 4)
        {
            "stem_idx": 4, "scale": 1.08, "open": 0.94,
            "rot": 0.28, "bud_t0": 3.9, "bloom_t0": 7.8, "stamen_t0": 10.8,
            "seed": 350, "dof": "mid"
        },
    ]

    for f in flowers_spec:
        pos_seq, tan_seq, _ = stem_data[f["stem_idx"]]
        flower_center = pos_seq[-1]
        flower_dir = normalize_v3(tan_seq[-1])
        basis = build_basis_from_dir(flower_dir)

        sc = f["scale"]
        dof = f["dof"]
        bud_t0 = f["bud_t0"]
        pt0 = f["bloom_t0"]
        st0 = f["stamen_t0"]
        base_rot = f["rot"]

        # 6 Tepals: 3 Outer Whorl + 3 Inner Whorl
        # Outer whorl at 0, 120, 240 deg
        for k in range(3):
            az = base_rot + k * (2.0 * math.pi / 3.0)
            p_delay = 0.22 * k
            elements.append(generate_tulip_tepal(
                center=flower_center, basis=basis, azimuth=az,
                scale=sc, is_inner=False, max_openness=f["open"],
                wire_t0=bud_t0 + 0.05 * k,
                wire_t1=bud_t0 + 0.95 + 0.05 * k,
                surf_t0=bud_t0 + 0.20 + 0.05 * k,
                surf_t1=bud_t0 + 1.25 + 0.05 * k,
                bloom_t0=pt0 + p_delay,
                bloom_t1=pt0 + p_delay + 2.5,
                rng_seed=f["seed"] + k,
                dof_layer=dof
            ))

        # Inner whorl at 60, 180, 300 deg
        for k in range(3):
            az = base_rot + math.pi / 3.0 + k * (2.0 * math.pi / 3.0)
            p_delay = 0.18 + 0.22 * k
            elements.append(generate_tulip_tepal(
                center=flower_center, basis=basis, azimuth=az,
                scale=sc, is_inner=True, max_openness=f["open"],
                wire_t0=bud_t0 + 0.10 + 0.05 * k,
                wire_t1=bud_t0 + 1.00 + 0.05 * k,
                surf_t0=bud_t0 + 0.25 + 0.05 * k,
                surf_t1=bud_t0 + 1.30 + 0.05 * k,
                bloom_t0=pt0 + p_delay,
                bloom_t1=pt0 + p_delay + 2.5,
                rng_seed=f["seed"] + 10 + k,
                dof_layer=dof
            ))

        # 6 Golden Stamens + 3-lobed Pistil
        elements.extend(generate_tulip_androecium_and_pistil(
            center=flower_center, basis=basis, scale=sc,
            wire_t0=st0 - 0.3, wire_t1=st0 + 1.0,
            surf_t0=st0, surf_t1=st0 + 1.5,
            bloom_t0=st0 + 0.1, bloom_t1=st0 + 2.0,
            rng_seed=f["seed"] + 50,
            dof_layer=dof
        ))

    return elements


# ============================================================================
# STUDIO BACKDROP, CAMERA PROJECTION & PBR/SSS SHADING
# ============================================================================
def create_tulip_studio_backdrop():
    """
    Deep midnight navy to indigo radial vignette (#050B1F center to #01030A edge).
    Preserves the same dark depth and vignette shape as reference.
    """
    y = np.linspace(-1.0, 1.0, HEIGHT, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32)[None, :]

    r2_center = (x * 0.90) ** 2 + ((y + 0.05) * 0.72) ** 2
    glow_main = np.exp(-1.55 * r2_center)

    r2_soft = ((x - 0.20) * 0.85) ** 2 + ((y + 0.25) * 0.80) ** 2
    glow_soft = np.exp(-2.1 * r2_soft)

    # In BGR:
    # Edge:   #01030A -> B: 10, G: 3,  R: 1
    # Center: #050B1F -> B: 31, G: 11, R: 5
    b = 8.0 + 24.0 * glow_main + 14.0 * glow_soft
    g = 3.0 + 10.0 * glow_main + 8.0 * glow_soft
    r = 1.5 + 4.5 * glow_main + 3.0 * glow_soft

    bg = np.stack([b, g, r], axis=-1)
    return np.clip(bg, 0, 255).astype(np.uint8)


STUDIO_BG = create_tulip_studio_backdrop()


def project_points(P_world, cam_rot, cam_target, cam_dist, fov_scale):
    P_rel = (P_world - cam_target) @ cam_rot.T
    z_depth = cam_dist - P_rel[..., 2]
    z_safe = np.maximum(z_depth, 0.25)
    sx = (WIDTH * 0.50) + (P_rel[..., 0] / z_safe) * fov_scale
    sy = (HEIGHT * 0.52) - (P_rel[..., 1] / z_safe) * fov_scale
    return sx, sy, z_depth, P_rel


def compute_lighting(normals_cam, base_rgb, sss_strength, spec_strength, shininess):
    """
    Physically-based 3D botanical shader:
      - Key light: cool ice-white
      - Fill light: faint warm golden fill for depth
      - Rim light: sapphire-cyan backlight + Subsurface Scattering for translucent tepals
      - Micro-cuticle Blinn-Phong specular highlight
      - Tone mapping preserving luminous blue gradients without blown-out clipping
    """
    L_key = normalize_v3(np.array([-0.38, 0.58, 0.72], dtype=np.float32))
    L_fill = normalize_v3(np.array([0.68, -0.18, 0.71], dtype=np.float32))
    L_rim = normalize_v3(np.array([0.15, 0.75, -0.64], dtype=np.float32))
    V_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    nz = normals_cam[..., 2:3]
    N = np.where(nz < 0.0, -normals_cam, normals_cam)

    ndotl_key = np.sum(N * L_key, axis=-1, keepdims=True)
    diff_key = np.clip((ndotl_key + 0.20) / 1.20, 0.0, 1.0)

    ndotl_fill = np.sum(N * L_fill, axis=-1, keepdims=True)
    diff_fill = np.clip((ndotl_fill + 0.35) / 1.35, 0.0, 1.0)

    ndotv = np.clip(np.sum(N * V_dir, axis=-1, keepdims=True), 0.0, 1.0)
    fresnel_rim = (1.0 - ndotv) ** 2.2
    back_trans = np.clip(np.sum(N * L_rim, axis=-1, keepdims=True) * 0.5 + 0.5, 0.0, 1.0)
    sss = sss_strength * (0.60 * fresnel_rim + 0.40 * back_trans)

    H_key = normalize_v3(L_key + V_dir)
    ndoth = np.clip(np.sum(N * H_key, axis=-1, keepdims=True), 0.0, 1.0)
    spec = spec_strength * (ndoth ** shininess)

    key_col = np.array([0.96, 0.98, 1.04], dtype=np.float32)
    fill_col = np.array([0.92, 0.84, 0.72], dtype=np.float32)
    sss_col = np.array([0.45, 0.75, 1.05], dtype=np.float32)   # Translucent cyan-blue SSS

    lit_rgb = (
        base_rgb * (0.16 + 0.70 * diff_key * key_col + 0.16 * diff_fill * fill_col)
        + base_rgb * sss * sss_col
        + spec * np.array([0.92, 0.96, 1.04], dtype=np.float32)
    )

    lit_rgb = np.clip(lit_rgb, 0.0, 1.25)
    lit_rgb = lit_rgb / (1.0 + 0.18 * lit_rgb) * 1.10
    return np.clip(lit_rgb, 0.0, 1.0)


# ============================================================================
# SINGLE-FRAME 3D RENDERER (WITH DEPTH OF FIELD & BOKEH)
# ============================================================================
SCENE_ELEMENTS = None


def init_worker(w=None, h=None, show_hud=False):
    global SCENE_ELEMENTS, WIDTH, HEIGHT, STUDIO_BG, SHOW_HUD
    if w is not None and h is not None:
        WIDTH, HEIGHT = w, h
        STUDIO_BG = create_tulip_studio_backdrop()
    SHOW_HUD = show_hud
    SCENE_ELEMENTS = build_tulip_scene()


def rasterize_layer(canvas, quad_pts_list, quad_z_list, quad_bgr_list, pollen_list):
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
    global SCENE_ELEMENTS
    if SCENE_ELEMENTS is None:
        SCENE_ELEMENTS = build_tulip_scene()

    t = (frame_idx / max(1, total_frames - 1)) * DURATION
    p_all = frame_idx / max(1, total_frames - 1)

    bg_fade = 0.35 + 0.65 * smootherstep(0.0, 1.5, t)
    canvas = (STUDIO_BG.astype(np.float32) * bg_fade).astype(np.uint8)

    # 1. Camera Trajectory (identical to reference)
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

    # 2. Evaluate & Bucket Active Elements
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

        if s_prog > 0.002:
            P_s, normals_s, c_quad_s, _ = slice_mesh_along_u(P_full, C_full, s_prog)
            normals_cam = normals_s @ cam_rot.T
            sx, sy, zd, _ = project_points(P_s, cam_rot, cam_target, cam_dist, fov_scale)

            lit_rgb = compute_lighting(normals_cam, c_quad_s, el.sss_strength, el.spec_strength, el.shininess)

            q_x = np.stack([sx[:-1, :-1], sx[1:, :-1], sx[1:, 1:], sx[:-1, 1:]], axis=-1)
            q_y = np.stack([sy[:-1, :-1], sy[1:, :-1], sy[1:, 1:], sy[:-1, 1:]], axis=-1)
            pts_int = np.rint(np.stack([q_x, q_y], axis=-1).reshape(-1, 4, 2) * SUBPIX).astype(np.int32)
            z_q = (0.25 * (zd[:-1, :-1] + zd[1:, :-1] + zd[1:, 1:] + zd[:-1, 1:])).reshape(-1)
            bgr_q = np.rint(lit_rgb[..., ::-1] * 255.0).astype(np.uint8).reshape(-1, 3)

            bucket = layers[el.dof_layer]
            bucket["pts"].append(pts_int)
            bucket["z"].append(z_q)
            bucket["bgr"].append(bgr_q)

            if el.pollen_tip is not None and s_prog > 0.60:
                pos_c, pos_o, rad_px, core_bgr, rim_bgr = el.pollen_tip
                tip_3d = pos_c * (1.0 - b_prog) + pos_o * b_prog
                if wind_vec is not None and el.anchor is not None:
                    dist_a = float(np.linalg.norm(tip_3d - el.anchor))
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

        base_wire_vis = 1.0 - 0.88 * smootherstep(0.55, 1.0, s_prog)
        if el.extra_wire_t0 is not None and el.extra_wire_t1 is not None:
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
                        float(wsx[-1, j]), float(wsy[-1, j]), (255, 215, 160), 2.5
                    ))

            for i in range(0, Nu_w, el.wire_stride_u):
                pts_2d = np.stack([wsx[i, :], wsy[i, :]], axis=-1)
                pts_sub = np.rint(pts_2d * SUBPIX).astype(np.int32)
                z_m = float(np.mean(wzd[i, :])) - 0.015
                wire_lines.append((pts_sub, z_m, wc_bgr, 1))

    # 3. Drifting Sapphire Tepals (Stage 5)
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

    # 4. Composite Depth-of-Field Layers
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

    # 5. Wireframe Overlay
    if wire_lines:
        wire_layer = np.zeros_like(canvas)
        wire_lines.sort(key=lambda item: item[1], reverse=True)
        for pts_sub, _, wc_bgr, thick in wire_lines:
            cv2.polylines(
                wire_layer, [pts_sub], isClosed=False, color=wc_bgr,
                thickness=thick, lineType=cv2.LINE_AA, shift=SHIFT
            )
        canvas = cv2.addWeighted(canvas, 1.0, wire_layer, 0.40, 0.0)

    # 6. Ambient Sparkles (Cool Ice-Blue / White + Warm Gold Pollen Specks)
    particle_layer = np.zeros_like(canvas)
    rng_p = np.random.RandomState(108)
    n_ambient = 62
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
            if i % 5 == 0:
                # Warm gold pollen speck
                col = (int(35 * twinkle), int(185 * twinkle), int(235 * twinkle))
            else:
                # Cool ice-cyan / sapphire sparkle
                col = (int(255 * twinkle), int(205 * twinkle), int(135 * twinkle))
            cv2.circle(particle_layer, (px, py), max(1, rad), col, -1, lineType=cv2.LINE_AA)

    for px, py, bgr, rad in wave_particles[:240]:
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < WIDTH and 0 <= iy < HEIGHT:
            cv2.circle(particle_layer, (ix, iy), max(1, int(round(rad))), bgr, -1, lineType=cv2.LINE_AA)

    particle_blur = cv2.GaussianBlur(particle_layer, (0, 0), sigmaX=3.5)
    canvas = cv2.addWeighted(canvas, 1.0, particle_blur, 0.80, 0.0)
    canvas = cv2.addWeighted(canvas, 1.0, particle_layer, 0.45, 0.0)

    # 7. Subtle Optical Highlight Bloom & Optional Telemetry HUD
    bright = np.maximum(canvas.astype(np.float32) - 210.0, 0.0).astype(np.uint8)
    bloom = cv2.GaussianBlur(bright, (0, 0), sigmaX=14.0)
    canvas = cv2.addWeighted(canvas, 1.0, bloom, 0.28, 0.0)

    if SHOW_HUD:
        draw_botanical_hud(canvas, t)

    return canvas


def draw_botanical_hud(canvas, t):
    hud_alpha = smootherstep(0.3, 1.2, t) * (1.0 - 0.40 * smootherstep(15.0, 16.8, t))
    if hud_alpha <= 0.02:
        return

    overlay = canvas.copy()
    c_main = (240, 230, 220)
    c_accent = (245, 195, 145)
    c_muted = (168, 155, 145)

    cv2.putText(
        overlay, "TULIPA  //  BLUE TULIP (SAPPHIRE CULTIVAR)", (68, 118),
        cv2.FONT_HERSHEY_SIMPLEX, 0.74, c_main, 2, cv2.LINE_AA
    )
    cv2.putText(
        overlay, "3D PROCEDURAL BOTANICAL SPECIMEN  //  6-TEPAL GOBLET ANTHESIS", (68, 152),
        cv2.FONT_HERSHEY_SIMPLEX, 0.41, c_muted, 1, cv2.LINE_AA
    )
    cv2.line(overlay, (68, 172), (450, 172), (85, 105, 125), 1, cv2.LINE_AA)

    if t < 3.0:
        stage_str = "STAGE 01 // CAULESCENT STEM & GLAUCOUS LEAF GROWTH"
        eq_str = "B(u,v) : Cubic Bezier Parallel Transport + Lanceolate Foliage"
    elif t < 6.0:
        stage_str = "STAGE 02 // OVOID TEARDROP BUD FORMATION & CYAN APEX TINT"
        eq_str = "H(u) : Closed 6-Tepal Imbrication + Glaucous Sheen"
    elif t < 9.0:
        stage_str = "STAGE 03 // 6-TEPAL GOBLET UNFOLDING & TRANSLUCENT SSS"
        eq_str = "P(u,v) : Outer/Inner Tepal Separation + Sapphire venation"
    elif t < 13.0:
        stage_str = "STAGE 04 // 6 GOLDEN STAMENS, ANTHERS & 3-LOBED STIGMA"
        eq_str = "Androecium Whorl + Golden Pollen Anthers + Pistil Maturation"
    else:
        stage_str = "STAGE 05 // CINEMATIC BLUE TULIP SPECIMEN & DRIFTING TEPALS"
        eq_str = "3D Depth-of-Field Showcase + Sapphire Petal Aerodynamics"

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
    cv2.rectangle(overlay, (68, y_bot + 46), (68 + bar_w, y_bot + 50), (45, 52, 65), -1)
    cv2.rectangle(
        overlay, (68, y_bot + 46), (68 + int(bar_w * prog), y_bot + 50),
        (245, 185, 125), -1
    )

    cv2.addWeighted(overlay, hud_alpha * 0.82, canvas, 1.0 - hud_alpha * 0.82, 0.0, dst=canvas)


# ============================================================================
# CLI & VIDEO ENCODING (MANDATORY ALL-INTRA H.264 FLAGS)
# ============================================================================
def find_ffmpeg():
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


def render_preview_keyframes(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    init_worker(WIDTH, HEIGHT, SHOW_HUD)
    test_times = [0.0, 3.0, 6.0, 9.0, 13.0, 18.0]
    aspect_tag = f"{WIDTH}x{HEIGHT}"
    for sec in test_times:
        f_idx = int(round((sec / DURATION) * (TOTAL_FRAMES - 1)))
        t0 = time.time()
        img = render_frame(f_idx, TOTAL_FRAMES)
        out_path = os.path.join(out_dir, f"preview_{aspect_tag}_{sec:.1f}s.png")
        cv2.imwrite(out_path, img)
        print(f"Rendered {out_path} in {time.time() - t0:.2f}s")


def render_full_video(output_mp4, max_frames=None):
    target_frames = min(TOTAL_FRAMES, max_frames) if max_frames else TOTAL_FRAMES
    print(f"Rendering {target_frames}/{TOTAL_FRAMES} frames at {WIDTH}x{HEIGHT} ({FPS} fps, {DURATION:.1f}s)...", flush=True)
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
            "-crf", "18",
            "-g", "1",
            "-bf", "0",
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
            else:
                cv_writer.write(frame)

            if (idx + 1) % 30 == 0 or (idx + 1) == target_frames:
                elapsed = time.time() - t_start
                fps_render = (idx + 1) / max(1e-4, elapsed)
                eta = (target_frames - idx - 1) / max(1e-4, fps_render)
                print(f"[{idx + 1:3d}/{target_frames}] {fps_render:5.1f} fps | Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s", flush=True)

    if proc is not None:
        stdout_data, stderr_data = proc.communicate()
        stderr_out = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
        if proc.returncode != 0:
            print("FFmpeg stderr:\n", stderr_out, flush=True)
            raise RuntimeError(f"FFmpeg encoding failed with exit code {proc.returncode}")
    else:
        cv_writer.release()

    size_mb = os.path.getsize(output_mp4) / (1024.0 * 1024.0)
    print(f"Render complete: {output_mp4} ({size_mb:.2f} MB in {time.time() - t_start:.1f}s)", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Procedural 3D Blue Tulip Botanical Animation Renderer")
    parser.add_argument("--aspect", choices=["portrait", "landscape"], default="landscape",
                        help="Render aspect ratio: landscape (1920x1080) or portrait (1080x1920)")
    parser.add_argument("--preview", action="store_true", help="Render preview keyframe images to ./previews/")
    parser.add_argument("--frames", type=int, default=None, help="Render only first N frames")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 filename")
    parser.add_argument("--hud", action="store_true", help="Draw botanical specimen telemetry overlay")
    args = parser.parse_args()

    if args.aspect == "portrait":
        WIDTH, HEIGHT = 1080, 1920
    else:
        WIDTH, HEIGHT = 1920, 1080

    SHOW_HUD = args.hud

    if args.preview:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "previews")
        render_preview_keyframes(out_dir)
    else:
        out_name = args.out or (f"tulip_{args.aspect}.mp4")
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_name)
        render_full_video(out_path, max_frames=args.frames)
