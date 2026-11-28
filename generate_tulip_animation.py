#!/usr/bin/env python3
"""
Cinematic 3D Blue Tulip (Tulipa, Stylized Sapphire Cultivar) Botanical Animation (v2.4.0)
========================================================================================
High-Fidelity Procedural 3D Botanical Model & Look-Dev:
  - 18 Flower Heads Total (15 Open Layered Blooms + 3 Closed 6-Tepal Imbricate Buds)
  - 14 Caulescent Stems + 18 Coaxial Peduncle Necks with receptacle domes
  - 29 Broad Glaucous Leaves with V-crease, S-curves & powder bloom Fresnel
  - 9 Tepals per Open Flower (3 Outer, 3 Inner, 3 Ruffled Petaloids) via pre-solved K=14 P_states unfolding
  - Two-sided sapphire pigment (royal/cobalt body, azure/cyan ridges, indigo folds, pale rim, warm inner base glow)
  - 6 Golden Bilobed Anthers with orange-gold pollen beads, golden filaments & slimmer 3-lobed pistil
  - Physically-based botanical shader with azure-leaning SSS tint & seam-free quads (0.55px expansion)
  - Light studio backdrop (ivory default, pearl, champagne) with soft cast bouquet shadow & highlight bloom
  - 3-Layer Depth-of-Field (bg, mid, fg) + drifting two-sided sapphire tepals + golden pollen shimmer
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
    y_axis = direction / max(1e-8, float(np.linalg.norm(direction)))
    if abs(float(np.dot(y_axis, up_hint))) > 0.92:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_axis = np.cross(up_hint, y_axis)
    x_axis /= max(1e-8, float(np.linalg.norm(x_axis)))
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(1e-8, float(np.linalg.norm(z_axis)))
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
    if abs(float(np.dot(t0, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    n1 = np.cross(t0, ref)
    n1 /= max(1e-8, float(np.linalg.norm(n1)))
    n2 = np.cross(t0, n1)
    N1[0], N2[0] = n1, n2

    for i in range(1, Nu):
        ti = T[i]
        n1_proj = n1 - np.dot(n1, ti) * ti
        norm_n1 = float(np.linalg.norm(n1_proj))
        if norm_n1 < 1e-6:
            n1_proj = np.cross(ti, n2)
            norm_n1 = float(np.linalg.norm(n1_proj))
        n1 = n1_proj / max(1e-8, norm_n1)
        n2 = np.cross(ti, n1)
        N1[i], N2[i] = n1, n2
    return N1, N2


# ============================================================================
# 3D BOTANICAL SURFACE ELEMENT
# ============================================================================
class BotanicalElement:
    def __init__(self, kind, P, C, wire_t0, wire_t1, surf_t0, surf_t1,
                 sss_strength=0.38, spec_strength=0.22, shininess=20.0,
                 wire_color=(0.35, 0.65, 0.95), wire_stride_u=4, wire_stride_v=3,
                 P_closed=None, C_closed=None, bloom_t0=None, bloom_t1=None,
                 extra_wire_t0=None, extra_wire_t1=None,
                 anchor=None, scale_t0=None, scale_t1=None, scale_init=1.0,
                 dof_layer="mid", pollen_tip=None, head_nod_phase=0.0, P_states=None):
        self.kind = kind
        # P_states: (K,Nu,Nv,3) pre-solved poses for continuous, length-preserving unfolding (K=1 -> fixed pose)
        self.P_states = P_states.astype(np.float32) if P_states is not None else None
        if self.P_states is not None:
            P = self.P_states[-1]
            P_closed = None
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
        self.dof_layer = dof_layer
        self.pollen_tip = pollen_tip
        self.head_nod_phase = head_nod_phase

    def evaluate_mesh(self, t, wind_vec=None):
        if self.P_states is not None:
            b_prog = smootherstep(self.bloom_t0, self.bloom_t1, t)
            K = self.P_states.shape[0]
            if K == 1:
                P = self.P_states[0].copy()
            else:
                s = b_prog * (K - 1)
                i0 = min(int(s), K - 2)
                f = s - i0
                P = self.P_states[i0] * (1.0 - f) + self.P_states[i0 + 1] * f
        elif self.P_closed is not None:
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

        # Organic head wind nod (2-3 degrees sine with per-flower phase from 9s)
        if self.kind in ("petal", "stamen", "bud") and self.anchor is not None and t > 9.0:
            nod_amp = 0.045 * smootherstep(9.0, 13.0, t)
            nod_ang = nod_amp * math.sin((t - 9.0) * 1.85 + self.head_nod_phase)
            P_rel = P - self.anchor
            c_nod, s_nod = math.cos(nod_ang), math.sin(nod_ang)
            # Nod in Z-Y
            y_nod = P_rel[..., 1] * c_nod - P_rel[..., 2] * s_nod
            z_nod = P_rel[..., 1] * s_nod + P_rel[..., 2] * c_nod
            P = self.anchor + np.stack([P_rel[..., 0], y_nod, z_nod], axis=-1)

        if wind_vec is not None and self.anchor is not None and self.kind in ("petal", "leaf", "stamen", "stem"):
            dist_from_anchor = np.linalg.norm(P - self.anchor, axis=-1, keepdims=True)
            sway_factor = (dist_from_anchor ** 1.35) * (0.28 if self.kind == "petal" else 0.22)
            P = P + sway_factor * wind_vec

        return P, C, b_prog


def slice_mesh_along_u(P, C, prog):
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
                        rng_seed=42, dof_layer="mid", cap_radius=None, cap_len=0.0):
    """
    Generate a slender, waxy curved green tulip stem along a cubic Bezier spline
    with Bishop transport frame and realistic peduncle neck swelling.
    """
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    pos, T = eval_cubic_bezier(p0, p1, p2, p3, u)
    T = normalize_v3(T)
    N1, N2 = compute_bishop_frame(T)

    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    theta = (v + 1.0) * math.pi

    P = np.zeros((Nu, Nv, 3), dtype=np.float32)
    C = np.zeros((Nu, Nv, 3), dtype=np.float32)

    # Stem color: #34693F base -> #7FA84F tip with longitudinal sheen stripe
    c_stem_base = np.array([0.20, 0.41, 0.25], dtype=np.float32)   # #34693F
    c_stem_mid = np.array([0.32, 0.52, 0.28], dtype=np.float32)
    c_stem_top = np.array([0.50, 0.66, 0.31], dtype=np.float32)   # #7FA84F

    seg_len = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    path_len = float(s_cum[-1])
    c_cap = np.array([0.42, 0.62, 0.40], dtype=np.float32)

    for i in range(Nu):
        ui = u[i]
        # Neck swelling in last 12% before receptacle
        neck_swell = 1.0 + 0.16 * smoothstep(0.85, 0.98, ui) * (1.0 - smoothstep(0.98, 1.0, ui))
        r = (r_base * (1.0 - ui) + r_tip * ui) * neck_swell
        cap_w = 0.0
        if cap_radius is not None:
            # Receptacle bulb: the peduncle flares into the flower base and closes in a dome INSIDE the tepal
            # whorl, so the stem is one continuous solid body with the flower (no open ends, no gaps).
            d_end = path_len - float(s_cum[i])
            dome = 0.30 * cap_len
            if d_end < dome:
                r = cap_radius * math.sqrt(max(0.0, 1.0 - ((dome - d_end) / dome) ** 2))
                cap_w = 1.0
            else:
                cap_w = float(smoothstep(cap_len, dome, d_end))
                r = r + (cap_radius - r) * cap_w
        P[i] = pos[i] + r * (np.cos(theta)[:, None] * N1[i] + np.sin(theta)[:, None] * N2[i])

        if ui < 0.5:
            mix = ui / 0.5
            col = (1.0 - mix) * c_stem_base + mix * c_stem_mid
        else:
            mix = (ui - 0.5) / 0.5
            col = (1.0 - mix) * c_stem_mid + mix * c_stem_top
        sheen = 0.97 + 0.03 * np.cos(4.0 * theta)[:, None]
        C[i] = col * sheen
        if cap_w > 0.0:
            C[i] = C[i] * (1.0 - 0.35 * cap_w) + c_cap * (0.35 * cap_w)

    el = BotanicalElement(
        kind="stem", P=P, C=C,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.22, spec_strength=0.28, shininess=28.0,
        wire_color=(0.42, 0.78, 0.45), wire_stride_u=5, wire_stride_v=4,
        dof_layer=dof_layer
    )
    return el, pos, T




# Layer definitions (units of flower scale S). cb/ob = (base, mid, tip) bend angles [rad] of the
# midline from the flower axis for the CLOSED / fully OPEN pose; curl = extra tip reflex (open).
TEPAL_LAYERS = {
    "outer":    dict(L=0.54, wmax=0.205, rb=0.030, y0=-0.070, cb=(0.34, 0.21, -0.69), ob=(0.26, 0.62, 0.92),
                     curl=0.26, alpha_o=(1.30, 0.86, 0.42), ruf=0.026, flare=(0.88, 1.16), layer_k=0),
    "inner":    dict(L=0.51, wmax=0.215, rb=0.026, y0=-0.060, cb=(0.54, 0.06, -0.60), ob=(0.22, 0.48, 0.74),
                     curl=0.14, alpha_o=(1.30, 0.92, 0.50), ruf=0.022, flare=(0.85, 1.12), layer_k=1),
    "petaloid": dict(L=0.26, wmax=0.105, rb=0.020, y0=-0.036, cb=(0.60, 0.02, -0.50), ob=(0.06, 0.12, 0.32),
                     curl=-0.10, alpha_o=(1.30, 1.00, 0.70), ruf=0.020, flare=(0.80, 1.00), layer_k=2),
}

def _ss(a, b, x):
    t = np.clip((x - a) / max(1e-8, (b - a)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def tepal_planform(spec, U):
    s = np.maximum(0.0, np.sin(math.pi * np.clip(U, 0, 1) ** 0.80)) ** 0.46
    taper = 1.0 - 0.16 * _ss(0.82, 1.0, U) ** 1.5
    base = 0.26 * (1.0 - _ss(0.0, 0.22, U))
    return spec["wmax"] * (s * taper + base)

def tepal_midline(spec, o_eff, Un):
    """Integrate bend angle phi(U) along the tepal -> (r, y, phi) arrays; o_eff may exceed 1 (extra flare)."""
    L = spec["L"]
    t = float(np.clip(o_eff, 0.0, 1.6))
    pb = spec["cb"][0] + (spec["ob"][0] - spec["cb"][0]) * t
    pm = spec["cb"][1] + (spec["ob"][1] - spec["cb"][1]) * t
    pt = spec["cb"][2] + (spec["ob"][2] - spec["cb"][2]) * t
    s1 = _ss(0.0, 0.5, Un); s2 = _ss(0.30, 1.0, Un) ** 1.15; s3 = _ss(0.72, 1.0, Un)
    phi = pb + (pm - pb) * s1 + (pt - pm) * s2 + spec["curl"] * min(t, 1.0) * s3
    ds = L / (len(Un) - 1)
    pmid = 0.5 * (phi[1:] + phi[:-1])
    r = spec["rb"] + np.concatenate([[0.0], np.cumsum(ds * np.sin(pmid))])
    y = spec["y0"] + np.concatenate([[0.0], np.cumsum(ds * np.cos(pmid))])
    return r, y, phi


def tepal_surface_local(spec, o, rnd, Nu, Nv):
    """Return tepal vertices in the flower-local frame (y = flower axis) for open-amount `o` (0..1)."""
    U1 = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    V1 = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(U1, V1, indexing="ij")
    o_eff = o * rnd["flare"]
    oc = float(np.clip(o_eff, 0.0, 1.0))
    r, y, phi = tepal_midline(spec, o_eff, U1.astype(np.float64))
    r = r[:, None].astype(np.float32); y = y[:, None].astype(np.float32); phi = phi[:, None].astype(np.float32)
    az = spec["az"] + rnd["az_jit"] + rnd["lean_az"] * oc * U      # petals drift slightly off-radial as they open
    S = spec["S"]
    w = tepal_planform(spec, U) * rnd["wid"]
    w_safe = np.maximum(w, 1e-3)

    # --- cross-section: wraps around the axis when closed (alpha = w/r), relaxes to a gentle cup when open
    alpha_c = np.clip(w / np.maximum(r, 0.02), 0.35, 1.50)
    a0, a1, a2 = spec["alpha_o"]
    alpha_o = (a0 * (1 - _ss(0, 0.25, U)) + (a1 - (a1 - a2) * _ss(0.3, 1.0, U)) * _ss(0, 0.25, U)) * rnd["cup"]
    alpha = alpha_c * (1 - oc) + alpha_o * oc
    kappa = alpha / w_safe
    x = V * w * (1.0 + rnd["skew"] * V)                                  # lateral coordinate with left/right asymmetry
    ang = kappa * x
    lat = np.sin(ang) / np.maximum(kappa, 1e-4)
    nrm = -(1.0 - np.cos(ang)) / np.maximum(kappa, 1e-4)                 # inward (toward axis)

    absV = np.abs(V)
    mtip = _ss(0.35, 1.0, U)
    medge = _ss(0.45, 1.0, absV)
    # imbricate overlap: one margin lifts outward, the other tucks inward (same chirality for the whole whorl)
    nrm = nrm + 0.014 * (1.0 - 0.55 * oc) * V * (1.0 - 0.5 * _ss(0.7, 1.0, U))
    # keel (midrib ridge) + two soft secondary ridges: delicate longitudinal relief
    nrm = nrm + 0.0075 * np.exp(-(V / 0.17) ** 2) * (1.0 - 0.45 * U) \
              + 0.0030 * np.exp(-((absV - 0.52) / 0.12) ** 2) * np.sin(math.pi * np.clip(U, 0, 1))
    # margin ruffles (open flowers): undulation along the length plus a rolled-back rim
    ruf = spec["ruf"] * rnd["ruf"] * oc
    ph = rnd["ph"]
    wave = np.sin(2.0 * math.pi * (3.7 * U) + ph[0] + 2.2 * V) + 0.55 * np.sin(2.0 * math.pi * (6.1 * U) + ph[1] - 3.1 * V)
    nrm = nrm + ruf * medge * mtip * wave
    nrm = nrm + 0.060 * oc * (absV ** 3) * _ss(0.50, 1.0, U) * rnd["roll"]      # rim rolls outward
    nrm = nrm + 0.022 * oc * np.sin(2.0 * math.pi * (0.75 * U + 0.30 * V) + ph[2]) * rnd["und"]  # slow organic undulation
    lat = lat + 0.18 * ruf * medge * mtip * np.cos(2.0 * math.pi * (3.7 * U) + ph[0] + 2.2 * V)   # wavy outline
    # roll twist
    tw = rnd["twist"] * oc * U ** 1.5
    lat2 = lat * np.cos(tw) - nrm * np.sin(tw)
    nrm2 = lat * np.sin(tw) + nrm * np.cos(tw)

    ca, sa = np.cos(az), np.sin(az)
    er = np.stack([ca, np.zeros_like(ca), sa], axis=-1)
    el = np.stack([-sa, np.zeros_like(sa), ca], axis=-1)
    yy = np.zeros_like(er); yy[..., 1] = 1.0
    sphi, cphi = np.sin(phi)[..., None], np.cos(phi)[..., None]
    n_out = cphi * er - sphi * yy
    M = r[..., None] * er + y[..., None] * yy
    P = M + lat2[..., None] * el + nrm2[..., None] * n_out
    return (P * S).astype(np.float32), U, V


# ---------------- colour model (two-sided: outer = seen on buds / petal backs, inner = seen when open) -----------
PAL = dict(
    indigo=np.array([0.050, 0.075, 0.440], np.float32), cobalt=np.array([0.040, 0.250, 0.670], np.float32),
    royal=np.array([0.075, 0.400, 0.870], np.float32), azure=np.array([0.075, 0.600, 0.960], np.float32),
    cyan=np.array([0.220, 0.760, 0.990], np.float32), pale=np.array([0.520, 0.760, 0.960], np.float32),
    violet=np.array([0.250, 0.190, 0.740], np.float32), green=np.array([0.250, 0.500, 0.380], np.float32),
    warm=np.array([0.800, 0.560, 0.240], np.float32),
)

def _mix(a, b, w):
    return a * (1.0 - w[..., None]) + b * w[..., None]

def _noise2(U, V, ph, f=(1.0, 1.0)):
    return (0.5 * np.sin(U * 3.3 * f[0] + V * 2.1 * f[1] + ph[0]) + 0.5 * np.sin(U * 6.7 * f[0] - V * 3.7 * f[1] + ph[1]))

def tepal_colors(U, V, layer_k, ph, tone):
    """Return (C_outer_look, C_inner_look) for the tepal grid; both (Nu,Nv,3)."""
    absV = np.abs(V)
    n1 = _noise2(U, V, ph[:2]); n2 = _noise2(U, V, ph[2:4], (1.7, 1.3))
    # fine longitudinal striations (hair lines), converging toward the base, with slow modulation
    vv = V * (1.0 + 0.45 * (1.0 - U))
    st = (0.5 + 0.5 * np.sin(vv * 38.0 + 2.2 * np.sin(U * 5.0 + ph[4]) + 1.3 * n1)) * 0.6 \
       + (0.5 + 0.5 * np.sin(vv * 23.0 + 1.8 * np.sin(U * 3.2 + ph[5]))) * 0.4
    st = st ** 2.2
    fine = 0.5 + 0.5 * np.sin(vv * 71.0 + 3.0 * n2)
    tone_c = max(0.0, tone); tone_v = max(0.0, -tone)

    # ---------- OUTER surface ----------
    g = _ss(0.02, 0.55, U)
    body = _mix(PAL["cobalt"], PAL["royal"], g)
    ridge = np.exp(-((V - 0.22 * np.sin(U * 2.3 + ph[0])) / 0.30) ** 2) * _ss(0.10, 0.55, U) * (1.0 - 0.6 * _ss(0.85, 1.0, U))
    body = _mix(body, PAL["azure"], np.clip(0.70 * ridge * (0.7 + 0.3 * n1) + 0.18 * _ss(0.3, 0.9, U) * (0.5 + 0.5 * n2), 0, 1))
    body = _mix(body, PAL["cyan"], np.clip(0.55 * ridge * st * _ss(0.15, 0.7, U), 0, 1))
    body = _mix(body, PAL["indigo"], np.clip(0.42 * _ss(0.72, 1.0, absV) * (1.0 - 0.5 * _ss(0.6, 1.0, U)) + 0.10 * (1.0 - _ss(0.0, 0.18, U)), 0, 0.6))
    body = _mix(body, PAL["violet"], np.clip(0.20 * (0.5 + 0.5 * n2) * _ss(0.2, 0.7, U) + 0.30 * tone_v, 0, 0.5))
    body = _mix(body, PAL["cyan"], np.full_like(U, 0.20 * tone_c) * _ss(0.2, 0.9, U))
    body = body * (1.0 - 0.30 * (1.0 - st)[..., None] * (0.5 + 0.5 * _ss(0.05, 0.4, U))[..., None])  # dark hair-lines
    body = body * (1.0 + 0.12 * (fine - 0.5)[..., None])
    body = _mix(body, PAL["green"], 0.38 * (1.0 - _ss(0.0, 0.13, U)))                      # green-white basal flush
    body = _mix(body, PAL["pale"], np.clip(0.30 * _ss(0.82, 1.0, absV) * _ss(0.4, 1.0, U), 0, 0.5))   # translucent margin
    body = _mix(body, PAL["pale"], np.clip(0.12 * _ss(0.93, 1.0, absV) + 0.10 * _ss(0.94, 1.0, U), 0, 0.5))
    C_out = body

    # ---------- INNER surface ----------
    gi = _ss(0.02, 0.40, U)
    inn = _mix(PAL["indigo"], PAL["cobalt"], gi)
    inn = _mix(inn, PAL["royal"], _ss(0.30, 0.75, U) * (0.75 + 0.25 * n1))
    inn = _mix(inn, PAL["azure"], np.clip(0.46 * _ss(0.50, 1.0, U) * (0.45 + 0.55 * absV) + 0.30 * ridge * (0.5 + 0.5 * n2), 0, 0.8))
    inn = _mix(inn, PAL["cyan"], np.clip(0.30 * _ss(0.55, 1.0, U) * st * _ss(0.15, 0.9, absV) + 0.20 * tone_c, 0, 0.6))
    inn = _mix(inn, PAL["violet"], np.clip(0.34 * (0.5 + 0.5 * n2) * _ss(0.1, 0.55, U) * (1.0 - _ss(0.7, 1.0, U)) + 0.30 * tone_v, 0, 0.7))
    inn = inn * (1.0 - 0.32 * (1.0 - st)[..., None] * (0.5 + 0.5 * _ss(0.05, 0.5, U))[..., None])
    inn = inn * (1.0 + 0.13 * (fine - 0.5)[..., None])
    midrib = np.exp(-(V / 0.055) ** 2) * _ss(0.08, 0.30, U) * (1.0 - _ss(0.75, 1.0, U))
    inn = _mix(inn, PAL["indigo"], np.clip(0.30 * midrib, 0, 0.5))
    inn = _mix(inn, PAL["warm"], np.clip(0.40 * np.exp(-((U - 0.03) / 0.07) ** 2) * np.exp(-(V / 0.55) ** 2), 0, 0.45))   # warm basal glow
    inn = _mix(inn, PAL["pale"], np.clip(0.34 * _ss(0.80, 1.0, absV) * _ss(0.35, 1.0, U), 0, 0.5))
    inn = _mix(inn, PAL["pale"], np.clip(0.12 * _ss(0.93, 1.0, absV) + 0.10 * _ss(0.94, 1.0, U), 0, 0.5))
    C_in = inn
    if layer_k == 2:    # petaloids: deeper, more violet-indigo, frilly
        C_in = _mix(C_in, PAL["violet"], np.full_like(U, 0.22)); C_out = _mix(C_out, PAL["violet"], np.full_like(U, 0.16))
    return np.clip(C_out, 0, 1).astype(np.float32), np.clip(C_in, 0, 1).astype(np.float32)


def generate_tulip_tepal(center, basis, azimuth, scale, layer, max_openness,
                         wire_t0, wire_t1, surf_t0, surf_t1, bloom_t0, bloom_t1,
                         rng_seed=100, dof_layer="mid", head_nod_phase=0.0, tone=0.0, Nu=None, Nv=None):
    """
    One tepal of the sapphire tulip head (layer: "outer" | "inner" | "petaloid").

    The tepal is solved as a bend-angle midline (arc-length preserving) wrapped across its width, then
    pre-computed at K opening poses; render time interpolates between neighbouring poses, so unfolding is a
    continuous pivot about the receptacle with no vertex-lerp shrinkage. Closed pose = imbricate egg-shaped
    goblet bud; open pose = deep cup with rolled, ruffled, asymmetric margins.
    """
    rng = np.random.RandomState(rng_seed)
    spec = dict(TEPAL_LAYERS[layer])
    spec["az"] = azimuth
    spec["S"] = scale
    lo, hi = spec["flare"]
    L_var = rng.uniform(0.96, 1.04)
    spec["L"] = spec["L"] * L_var
    rnd = dict(
        flare=rng.uniform(lo, hi), az_jit=rng.uniform(-0.05, 0.05), lean_az=rng.uniform(-0.18, 0.18),
        wid=rng.uniform(0.96, 1.05), cup=rng.uniform(0.85, 1.12), skew=rng.uniform(-0.10, 0.10),
        ruf=rng.uniform(0.75, 1.30), ph=rng.uniform(0.0, 6.28, size=3), roll=rng.uniform(0.6, 1.4),
        und=rng.uniform(0.5, 1.4), twist=rng.uniform(-0.18, 0.18),
    )
    if Nu is None:
        Nu = 30 if layer == "petaloid" else 52
    if Nv is None:
        Nv = 33 if layer == "petaloid" else 55
    K = 1 if max_openness <= 0.001 else 14
    states = []
    for k in range(K):
        o = 0.0 if K == 1 else max_openness * k / (K - 1)
        P_loc, U, V = tepal_surface_local(spec, o, rnd, Nu, Nv)
        states.append(P_loc)
    P_loc_states = np.stack(states, axis=0)
    ph8 = rng.uniform(0.0, 6.28, size=8)
    C_out, C_in = tepal_colors(U, V, spec["layer_k"], ph8, tone)

    # Orientation: make the surface normal (d_v x d_u) point away from the flower axis so that "outer" colours
    # are shown on the outside of the whorl and "inner" colours on the inside, whichever way the camera looks.
    Pc = P_loc_states[0]
    i, j = Nu // 2, Nv // 2
    nrm = np.cross(Pc[i, j + 1] - Pc[i, j], Pc[i + 1, j] - Pc[i, j])
    radial = np.array([math.cos(azimuth), 0.0, math.sin(azimuth)], dtype=np.float32)
    if float(np.dot(nrm, radial)) < 0.0:
        P_loc_states = P_loc_states[:, :, ::-1]
        C_out = C_out[:, ::-1]
        C_in = C_in[:, ::-1]
    P_states = center + P_loc_states @ basis.T
    C = np.concatenate([C_out, C_in], axis=-1)

    return BotanicalElement(
        kind="petal", P=P_states[-1], P_states=P_states, C=C,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        bloom_t0=bloom_t0 if K > 1 else 1.0e9, bloom_t1=bloom_t1 if K > 1 else 1.0e9 + 1.0,
        extra_wire_t0=(bloom_t0 - 0.2) if K > 1 else None, extra_wire_t1=bloom_t1 if K > 1 else None,
        sss_strength=0.34, spec_strength=0.17, shininess=26.0,
        wire_color=(0.28, 0.52, 0.95), wire_stride_u=6, wire_stride_v=6,
        anchor=center, scale_t0=surf_t0, scale_t1=surf_t1 + 0.5, scale_init=0.35,
        dof_layer=dof_layer, head_nod_phase=head_nod_phase
    )


def generate_tulip_androecium_and_pistil(center, basis, scale, wire_t0, wire_t1, surf_t0, surf_t1,
                                         bloom_t0, bloom_t1, rng_seed=500, dof_layer="mid",
                                         head_nod_phase=0.0):
    """
    Generate the internal reproductive anatomy:
      - 6 anthers as bilobed elongated capsules, orange-gold core with deeper amber rim.
      - Golden filaments flaring 14-26 deg.
      - 1 central pale green-white ovary #CFE8C8 with 3-lobed stigma.
    """
    rng = np.random.RandomState(rng_seed)
    elements = []

    Nu, Nv = 20, 10
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    theta = (v + 1.0) * math.pi

    # 1. Central 3-Lobed Pistil
    p_len = 0.135 * scale
    p_pos = np.stack([np.zeros_like(u), p_len * u, np.zeros_like(u)], axis=-1)
    P_pistil = np.zeros((Nu, Nv, 3), dtype=np.float32)
    C_pistil = np.zeros((Nu, Nv, 3), dtype=np.float32)

    c_ovary_base = np.array([0.52, 0.76, 0.54], dtype=np.float32)
    c_stigma = np.array([0.81, 0.91, 0.78], dtype=np.float32)  # #CFE8C8

    for i in range(Nu):
        ui = u[i]
        if ui < 0.75:
            # Triangular prismatic ovary
            r = scale * (0.0105 + 0.004 * ui) * (1.0 + 0.18 * math.cos(3.0 * theta[0]))
        else:
            # Recurved 3-lobed stigma crest
            st_u = (ui - 0.75) / 0.25
            r = scale * (0.013 + 0.017 * math.sin(st_u * math.pi))
        P_pistil[i] = p_pos[i] + np.stack([r * np.cos(theta), np.zeros_like(theta), r * np.sin(theta)], axis=-1)
        C_pistil[i] = (1.0 - ui) * c_ovary_base + ui * c_stigma

    P_pistil_world = center + P_pistil @ basis.T
    elements.append(BotanicalElement(
        kind="stamen", P=P_pistil_world, C=C_pistil,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.30, spec_strength=0.28, shininess=22.0,
        wire_color=(0.65, 0.90, 0.65), wire_stride_u=5, wire_stride_v=3,
        anchor=center, dof_layer=dof_layer, head_nod_phase=head_nod_phase
    ))

    # 2. 6 Golden Stamens with Golden Filaments & Bilobed Anther Capsules
    c_fil_base = np.array([0.74, 0.48, 0.10], dtype=np.float32)  # deep amber
    c_fil_top = np.array([0.98, 0.76, 0.16], dtype=np.float32)   # golden
    c_anther_core = (28, 160, 236)  # BGR warm orange-gold
    c_anther_rim = (8, 100, 176)    # BGR deeper amber rim

    stamen_count = 6
    s_len = 0.200 * scale
    r0 = 0.028 * scale

    for k in range(stamen_count):
        dly = 0.014 * k
        az = (k / float(stamen_count)) * 2.0 * math.pi + rng.uniform(-0.03, 0.03)
        flare_open = 0.35 + rng.uniform(0.0, 0.08)  # Flaring 14 to 26 degrees
        ca, sa = math.cos(az), math.sin(az)
        rad_dir = np.array([ca, 0.0, sa], dtype=np.float32)
        up_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        def make_filament(flare_a):
            q0 = r0 * rad_dir + 0.006 * scale * up_dir
            q1 = (r0 + 0.22 * s_len * math.sin(flare_a * 0.7)) * rad_dir + (0.35 * s_len * math.cos(flare_a * 0.7)) * up_dir
            q2 = (r0 + 0.58 * s_len * math.sin(flare_a)) * rad_dir + (0.72 * s_len * math.cos(flare_a)) * up_dir
            q3 = (r0 + 0.88 * s_len * math.sin(flare_a * 0.95)) * rad_dir + (1.02 * s_len * math.cos(flare_a * 0.90)) * up_dir
            pos_c, _ = eval_cubic_bezier(q0, q1, q2, q3, u)
            P_f = np.zeros((Nu, Nv, 3), dtype=np.float32)
            r_fil = 0.0060 * scale
            for i in range(Nu):
                rf = r_fil * (1.0 - 0.22 * u[i])
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
            max(3, int(round(3.6 * scale))),
            c_anther_core, c_anther_rim
        )

        elements.append(BotanicalElement(
            kind="stamen", P=P_f_open_w, P_closed=P_f_closed_w, C=C_f,
            wire_t0=wire_t0 + dly, wire_t1=wire_t1 + dly,
            surf_t0=surf_t0 + dly, surf_t1=surf_t1 + dly,
            bloom_t0=bloom_t0 + dly, bloom_t1=bloom_t1 + dly,
            sss_strength=0.32, spec_strength=0.35, shininess=22.0,
            wire_color=(0.95, 0.82, 0.30), wire_stride_u=5, wire_stride_v=3,
            anchor=center, dof_layer=dof_layer,
            pollen_tip=pollen_tip, head_nod_phase=head_nod_phase
        ))

    return elements


def generate_tulip_leaf(origin, direction, length, width, azimuth, tilt,
                        wire_t0, wire_t1, surf_t0, surf_t1,
                        rng_seed=200, dof_layer="mid"):
    """
    Generate an authentic broad lanceolate tulip leaf:
      - Width: 0.14 to 0.22, length 0.82 to 1.28
      - Folded V-cross section with central crease (depth 0.022)
      - Graceful S-curve arching outward
      - Glaucous blue-green waxy color with powder bloom Fresnel
    """
    Nu, Nv = 44, 25
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')

    # Broad lanceolate taper: widest at ~40% length, tapering smoothly
    w_prof = width * np.sin(math.pi * (U ** 0.65)) * (1.0 - 0.55 * U)

    def build_leaf(unfold_t):
        reach = length * (0.35 + 0.65 * unfold_t) * U
        # Graceful S-curve
        elev = length * (0.30 * U - 0.48 * (U ** 2.0) * (0.5 + 0.5 * unfold_t) + 0.12 * (U ** 3.0))
        # Central V-shaped fold (crease depth 0.022)
        v_fold = 0.022 * np.abs(V) * (1.0 - 0.25 * U) * (1.0 - 0.30 * unfold_t)

        x_loc = w_prof * V
        y_loc = reach
        z_loc = elev + v_fold

        P_loc = np.stack([x_loc, y_loc, z_loc], axis=-1)
        rot = rot_y(azimuth) @ rot_x(tilt)
        return origin + P_loc @ rot.T

    P_closed = build_leaf(0.25)
    P_open = build_leaf(1.00)

    # Foliage colors: #2A5C3C base to #86B467 tip + glaucous bloom
    c_base = np.array([0.16, 0.36, 0.24], dtype=np.float32)   # #2A5C3C
    c_mid = np.array([0.28, 0.48, 0.32], dtype=np.float32)
    c_tip = np.array([0.53, 0.71, 0.40], dtype=np.float32)   # #86B467
    c_bloom = np.array([0.80, 0.90, 0.95], dtype=np.float32) # Blue-white glaucous sheen

    u_col = np.where(U[..., None] < 0.5,
                     (1.0 - U[..., None] * 2.0) * c_base + (U[..., None] * 2.0) * c_mid,
                     (1.0 - (U[..., None] - 0.5) * 2.0) * c_mid + ((U[..., None] - 0.5) * 2.0) * c_tip)

    # Glaucous powdery sheen (Fresnel 0.18)
    glaucous_f = 0.18 * ((np.abs(V) ** 1.6)[..., None] + 0.4 * (U[..., None] ** 2))
    C = (1.0 - glaucous_f) * u_col + glaucous_f * c_bloom

    return BotanicalElement(
        kind="leaf", P=P_open, P_closed=P_closed, C=C,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        bloom_t0=surf_t0, bloom_t1=surf_t1 + 1.2,
        sss_strength=0.28, spec_strength=0.25, shininess=24.0,
        wire_color=(0.45, 0.78, 0.60), wire_stride_u=6, wire_stride_v=4,
        anchor=origin, scale_t0=surf_t0, scale_t1=surf_t1 + 0.9, scale_init=0.30,
        dof_layer=dof_layer
    )


# ============================================================================
# DRIFTING SAPPHIRE TEPAL TEMPLATE (STAGE 5 REVEAL)
# ============================================================================
def build_drifting_tepals_template():
    Nu, Nv = 30, 25
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')

    length = 0.42
    w_edge = 0.20 * tepal_planform({"wmax": 1.0}, U) * (1.0 - 0.15 * (V ** 2)) * V
    cup = -0.034 * (V ** 2) * np.sin(math.pi * U) + 0.05 * U ** 2 \
          + 0.012 * (np.abs(V) ** 2) * U * np.sin(U * 17.0 + V * 2.0)

    P = np.stack([w_edge, length * U, cup], axis=-1)

    # same sapphire pigment as the flower tepals (two-sided: concave side = inner look)
    C_out, C_in = tepal_colors(U, V, 0, np.array([0.7, 2.1, 3.3, 4.4, 1.2, 5.1, 0.0, 0.0]), 0.0)
    C = np.concatenate([C_in, C_out], axis=-1)

    rng = np.random.RandomState(88)
    specs = []
    # 28 Drifting tepals
    for i in range(28):
        specs.append({
            "start_pos": np.array([
                rng.uniform(-0.70, 0.80),
                rng.uniform(-0.90, 1.40),
                rng.uniform(-0.45, 0.60)
            ], dtype=np.float32),
            "vel": np.array([
                rng.uniform(-0.06, 0.08),
                rng.uniform(-0.16, -0.07),
                rng.uniform(-0.05, 0.06)
            ], dtype=np.float32),
            "scale": rng.uniform(0.65, 1.10),
            "rot_rates": np.array([rng.uniform(0.4, 1.1), rng.uniform(0.5, 1.3), rng.uniform(0.4, 1.0)]),
            "rot_phases": np.array([rng.uniform(0, 6.28), rng.uniform(0, 6.28), rng.uniform(0, 6.28)]),
            "t_spawn": 13.0 + rng.uniform(0.0, 3.2),
            "dof": "fg" if (i % 7 == 0) else ("bg" if (i % 5 == 0) else "mid")
        })

    return P, C, specs


DRIFT_P_BASE, DRIFT_C_BASE, DRIFT_SPECS = build_drifting_tepals_template()


# ============================================================================
# COMPLETE BLUE TULIP SCENE ASSEMBLY (18 HEADS ACROSS 5 STAGES)
# ============================================================================
def build_tulip_scene():
    """
    Construct the complete 3D Blue Tulip bouquet (v2.4.0):
      - 14 caulescent stems + 18 coaxial peduncle necks with receptacle domes
      - 29 broad glaucous leaves framing lower third and junctions
      - 18 Flower Heads Total:
          * 3 closed 6-tepal imbricate buds (Stems 7, 8, 13)
          * 15 blooming flowers (3 outer + 3 inner + 3 petaloid tepals = 9 tepals each)
          * 2 flowers in dof='bg', 1 in dof='fg', rest 'mid'
          * Hero flower faces camera (dir=[0.08, 0.22, 0.97], scale=1.28)
      - 6 golden bilobed stamens + central 3-lobed pistil per open flower
    """
    elements = []

    base_origin = np.array([-0.45, -2.15, -0.05], dtype=np.float32)

    # Stems: 14 stems with curved peduncle neck Beziers
    stems_spec = [
        # Stem 0: Primary Center Stem (Hero 1)
        {
            "p0": base_origin + np.array([0.02, 0.0, 0.0]),
            "p1": base_origin + np.array([0.12, 1.15, 0.10]),
            "p2": np.array([-0.05, -0.15, 0.16], dtype=np.float32),
            "p3": np.array([0.12, 0.70, 0.18], dtype=np.float32),
            "r_base": 0.052, "r_tip": 0.026,
            "wire_t0": 0.0, "wire_t1": 1.8, "surf_t0": 0.35, "surf_t1": 2.5,
            "dof": "mid", "seed": 11
        },
        # Stem 1: Lower-Center Hero Companion (Hero 2)
        {
            "p0": base_origin + np.array([-0.04, 0.0, 0.04]),
            "p1": base_origin + np.array([-0.18, 0.95, 0.14]),
            "p2": np.array([-0.36, -0.55, 0.22], dtype=np.float32),
            "p3": np.array([-0.24, 0.14, 0.26], dtype=np.float32),
            "r_base": 0.050, "r_tip": 0.026,
            "wire_t0": 0.3, "wire_t1": 2.0, "surf_t0": 0.65, "surf_t1": 2.7,
            "dof": "mid", "seed": 22
        },
        # Stem 2: Mid-Right Blossom Stem
        {
            "p0": base_origin + np.array([0.06, 0.0, -0.02]),
            "p1": base_origin + np.array([0.28, 0.88, 0.12]),
            "p2": np.array([0.38, -0.28, 0.20], dtype=np.float32),
            "p3": np.array([0.48, 0.36, 0.22], dtype=np.float32),
            "r_base": 0.048, "r_tip": 0.025,
            "wire_t0": 0.5, "wire_t1": 2.2, "surf_t0": 0.85, "surf_t1": 2.9,
            "dof": "mid", "seed": 33
        },
        # Stem 3: Lower-Left Branch Stem
        {
            "p0": base_origin + np.array([-0.08, 0.0, 0.06]),
            "p1": base_origin + np.array([-0.28, 0.72, 0.18]),
            "p2": np.array([-0.45, -0.75, 0.25], dtype=np.float32),
            "p3": np.array([-0.46, -0.28, 0.28], dtype=np.float32),
            "r_base": 0.046, "r_tip": 0.024,
            "wire_t0": 0.7, "wire_t1": 2.4, "surf_t0": 1.05, "surf_t1": 3.1,
            "dof": "mid", "seed": 44
        },
        # Stem 4: Upper-Right Crown Stem
        {
            "p0": base_origin + np.array([0.08, 0.0, -0.04]),
            "p1": base_origin + np.array([0.22, 1.35, 0.08]),
            "p2": np.array([0.26, 0.45, 0.14], dtype=np.float32),
            "p3": np.array([0.34, 1.35, 0.12], dtype=np.float32),
            "r_base": 0.046, "r_tip": 0.024,
            "wire_t0": 0.9, "wire_t1": 2.6, "surf_t0": 1.25, "surf_t1": 3.3,
            "dof": "mid", "seed": 55
        },
        # Stem 5: Upper-Left Crown Stem
        {
            "p0": base_origin + np.array([-0.02, 0.0, -0.02]),
            "p1": base_origin + np.array([-0.18, 1.25, 0.06]),
            "p2": np.array([-0.26, 0.50, 0.12], dtype=np.float32),
            "p3": np.array([-0.22, 1.18, 0.10], dtype=np.float32),
            "r_base": 0.044, "r_tip": 0.023,
            "wire_t0": 1.0, "wire_t1": 2.7, "surf_t0": 1.35, "surf_t1": 3.4,
            "dof": "mid", "seed": 66
        },
        # Stem 6: Foreground Left Specimen (dof='fg')
        {
            "p0": base_origin + np.array([-0.10, 0.0, 0.10]),
            "p1": base_origin + np.array([-0.35, 0.55, 0.25]),
            "p2": np.array([-0.52, -1.05, 0.35], dtype=np.float32),
            "p3": np.array([-0.55, -0.65, 0.40], dtype=np.float32),
            "r_base": 0.048, "r_tip": 0.025,
            "wire_t0": 1.1, "wire_t1": 2.8, "surf_t0": 1.45, "surf_t1": 3.4,
            "dof": "fg", "seed": 77
        },
        # Stem 7: Upper-Right Depth Bud Stem (Closed Bud 1, dof='mid')
        {
            "p0": base_origin + np.array([0.04, 0.0, -0.06]),
            "p1": base_origin + np.array([0.16, 1.25, 0.04]),
            "p2": np.array([0.18, 0.65, 0.08], dtype=np.float32),
            "p3": np.array([0.22, 1.45, 0.06], dtype=np.float32),
            "r_base": 0.038, "r_tip": 0.020,
            "wire_t0": 1.3, "wire_t1": 2.9, "surf_t0": 1.60, "surf_t1": 3.5,
            "dof": "mid", "seed": 88
        },
        # Stem 8: Background Depth Stem (Closed Bud 2, dof='bg')
        {
            "p0": base_origin + np.array([0.0, 0.0, -0.12]),
            "p1": base_origin + np.array([0.08, 0.95, -0.20]),
            "p2": np.array([0.04, 0.18, -0.26], dtype=np.float32),
            "p3": np.array([0.10, 0.88, -0.28], dtype=np.float32),
            "r_base": 0.036, "r_tip": 0.018,
            "wire_t0": 1.4, "wire_t1": 3.0, "surf_t0": 1.70, "surf_t1": 3.6,
            "dof": "bg", "seed": 99
        },
        # --- Bouquet fill (v3): 5 additional stems --------------------------------------
        # Stem 9: Upper-Left Outer Stem (open flower)
        {
            "p0": base_origin + np.array([-0.06, 0.0, 0.02]),
            "p1": base_origin + np.array([-0.40, 1.15, 0.10]),
            "p2": np.array([-0.62, 0.10, 0.14], dtype=np.float32),
            "p3": np.array([-0.68, 0.74, 0.10], dtype=np.float32),
            "r_base": 0.044, "r_tip": 0.023,
            "wire_t0": 0.8, "wire_t1": 2.5, "surf_t0": 1.15, "surf_t1": 3.2,
            "dof": "mid", "seed": 111
        },
        # Stem 10: Right Lower Outer Stem (open flower)
        {
            "p0": base_origin + np.array([0.07, 0.0, 0.0]),
            "p1": base_origin + np.array([0.46, 0.80, 0.10]),
            "p2": np.array([0.64, -0.45, 0.16], dtype=np.float32),
            "p3": np.array([0.72, 0.02, 0.12], dtype=np.float32),
            "r_base": 0.044, "r_tip": 0.023,
            "wire_t0": 0.9, "wire_t1": 2.6, "surf_t0": 1.25, "surf_t1": 3.3,
            "dof": "mid", "seed": 122
        },
        # Stem 11: Top Centre Crown Stem (late bloomer)
        {
            "p0": base_origin + np.array([0.0, 0.0, -0.03]),
            "p1": base_origin + np.array([0.02, 1.55, 0.0]),
            "p2": np.array([-0.10, 0.60, 0.02], dtype=np.float32),
            "p3": np.array([-0.06, 1.62, -0.04], dtype=np.float32),
            "r_base": 0.040, "r_tip": 0.021,
            "wire_t0": 1.2, "wire_t1": 2.9, "surf_t0": 1.55, "surf_t1": 3.5,
            "dof": "mid", "seed": 133
        },
        # Stem 12: Front Low Centre-Right Stem (partially open cup)
        {
            "p0": base_origin + np.array([0.09, 0.0, 0.08]),
            "p1": base_origin + np.array([0.36, 0.55, 0.30]),
            "p2": np.array([0.30, -1.00, 0.38], dtype=np.float32),
            "p3": np.array([0.34, -0.52, 0.42], dtype=np.float32),
            "r_base": 0.042, "r_tip": 0.022,
            "wire_t0": 1.5, "wire_t1": 3.0, "surf_t0": 1.85, "surf_t1": 3.6,
            "dof": "mid", "seed": 144
        },
        # Stem 13: Upper-Right Accent Bud Stem
        {
            "p0": base_origin + np.array([0.06, 0.0, -0.08]),
            "p1": base_origin + np.array([0.56, 1.10, -0.05]),
            "p2": np.array([0.60, 0.30, -0.06], dtype=np.float32),
            "p3": np.array([0.66, 1.12, -0.08], dtype=np.float32),
            "r_base": 0.034, "r_tip": 0.018,
            "wire_t0": 1.6, "wire_t1": 3.1, "surf_t0": 1.95, "surf_t1": 3.7,
            "dof": "mid", "seed": 155
        },
    ]

    # 15 Blooming Flowers
    flowers_spec = [
        # 1. Main Hero Center Flower (Stem 1: face-on straight to viewer z=0.97, scale 1.28)
        {
            "stem_idx": 1, "node_idx": -1, "offset": np.array([0.0, 0.06, 0.05]),
            "dir": np.array([0.08, 0.22, 0.97]), "scale": 1.28, "open": 1.00,
            "rot": 0.16, "ped_t0": 3.2, "petal_t0": 6.2, "stamen_t0": 10.0,
            "seed": 301, "dof": "mid", "nod_phase": 0.0
        },
        # 2. Hero Center-Right Blossom (Stem 0: face-on z=0.93, scale 1.18)
        {
            "stem_idx": 0, "node_idx": -1, "offset": np.array([0.04, 0.05, 0.04]),
            "dir": np.array([0.18, 0.30, 0.93]), "scale": 1.18, "open": 0.98,
            "rot": -0.24, "ped_t0": 3.4, "petal_t0": 6.5, "stamen_t0": 10.2,
            "seed": 310, "dof": "mid", "nod_phase": 0.6
        },
        # 3. Hero Lower-Left Branch Blossom (Stem 3: face-on z=0.92, scale 1.14)
        {
            "stem_idx": 3, "node_idx": -1, "offset": np.array([-0.02, 0.05, 0.04]),
            "dir": np.array([-0.28, 0.28, 0.92]), "scale": 1.14, "open": 0.96,
            "rot": 0.45, "ped_t0": 3.5, "petal_t0": 6.8, "stamen_t0": 10.4,
            "seed": 320, "dof": "mid", "nod_phase": 1.2
        },
        # 4. Hero Mid-Right Blossom (Stem 2: face-on z=0.91, scale 1.15)
        {
            "stem_idx": 2, "node_idx": -1, "offset": np.array([0.03, 0.06, 0.03]),
            "dir": np.array([0.32, 0.26, 0.91]), "scale": 1.15, "open": 0.98,
            "rot": -0.32, "ped_t0": 3.6, "petal_t0": 7.0, "stamen_t0": 10.5,
            "seed": 330, "dof": "mid", "nod_phase": 1.8
        },
        # 5. Lower-Center Spur Fork (Stem 1 mid: face-on z=0.98, scale 1.08)
        {
            "stem_idx": 1, "node_idx": 42, "offset": np.array([0.12, 0.14, 0.22]),
            "dir": np.array([0.15, -0.12, 0.98]), "scale": 1.08, "open": 0.95,
            "rot": 0.52, "ped_t0": 3.5, "petal_t0": 6.7, "stamen_t0": 10.3,
            "seed": 340, "dof": "mid", "nod_phase": 2.4
        },
        # 6. Upper-Left Crown Blossom (Stem 5: face-on z=0.91, scale 1.08)
        {
            "stem_idx": 5, "node_idx": -1, "offset": np.array([-0.02, 0.04, 0.03]),
            "dir": np.array([-0.26, 0.32, 0.91]), "scale": 1.08, "open": 0.94,
            "rot": 0.85, "ped_t0": 3.8, "petal_t0": 7.3, "stamen_t0": 10.8,
            "seed": 350, "dof": "mid", "nod_phase": 3.0
        },
        # 7. Lower-Left Outer Companion (Stem 3 mid: 3/4 view z=0.78, scale 1.04)
        {
            "stem_idx": 3, "node_idx": 44, "offset": np.array([-0.16, 0.08, 0.14]),
            "dir": np.array([-0.54, 0.32, 0.78]), "scale": 1.04, "open": 0.92,
            "rot": 1.10, "ped_t0": 3.7, "petal_t0": 7.1, "stamen_t0": 10.6,
            "seed": 360, "dof": "mid", "nod_phase": 3.6
        },
        # 8. Mid-Right Outer Companion (Stem 2 mid: 3/4 view z=0.82, scale 1.02)
        {
            "stem_idx": 2, "node_idx": 42, "offset": np.array([0.18, 0.08, 0.12]),
            "dir": np.array([0.54, 0.18, 0.82]), "scale": 1.02, "open": 0.90,
            "rot": 0.60, "ped_t0": 3.9, "petal_t0": 7.5, "stamen_t0": 10.9,
            "seed": 370, "dof": "mid", "nod_phase": 4.2
        },
        # 9. Top Crown Blossom (Stem 4: 3/4 view z=0.86, scale 1.06)
        {
            "stem_idx": 4, "node_idx": -1, "offset": np.array([0.04, 0.04, 0.02]),
            "dir": np.array([0.36, 0.35, 0.86]), "scale": 1.06, "open": 0.92,
            "rot": 0.15, "ped_t0": 4.0, "petal_t0": 7.7, "stamen_t0": 11.1,
            "seed": 380, "dof": "mid", "nod_phase": 4.8
        },
        # 10. Profile Goblet 1: Foreground Left Edge (Stem 6: z=0.42, scale 1.16, dof='fg')
        {
            "stem_idx": 6, "node_idx": -1, "offset": np.array([-0.05, 0.06, 0.06]),
            "dir": np.array([-0.82, 0.38, 0.42]), "scale": 1.16, "open": 0.88,
            "rot": 0.40, "ped_t0": 3.6, "petal_t0": 6.9, "stamen_t0": 10.5,
            "seed": 390, "dof": "fg", "nod_phase": 5.2
        },
        # 11. Profile Goblet 2: Background Right Depth (Stem 0 mid: z=0.48, scale 0.95, dof='bg')
        {
            "stem_idx": 0, "node_idx": 38, "offset": np.array([0.14, 0.10, -0.12]),
            "dir": np.array([0.76, 0.42, 0.48]), "scale": 0.95, "open": 0.88,
            "rot": -0.65, "ped_t0": 4.2, "petal_t0": 8.0, "stamen_t0": 11.3,
            "seed": 400, "dof": "bg", "nod_phase": 5.8
        },
        # 12. Upper-Left Outer Blossom (Stem 9: 3/4 view)
        {
            "stem_idx": 9, "node_idx": -1, "offset": np.array([-0.02, 0.05, 0.03]),
            "dir": np.array([-0.46, 0.30, 0.83]), "scale": 1.02, "open": 0.94,
            "rot": 0.30, "ped_t0": 3.7, "petal_t0": 7.2, "stamen_t0": 10.7,
            "seed": 410, "dof": "mid", "nod_phase": 0.9
        },
        # 13. Right Lower Outer Blossom (Stem 10: 3/4 view)
        {
            "stem_idx": 10, "node_idx": -1, "offset": np.array([0.02, 0.05, 0.03]),
            "dir": np.array([0.50, 0.12, 0.86]), "scale": 1.00, "open": 0.92,
            "rot": -0.50, "ped_t0": 3.9, "petal_t0": 7.6, "stamen_t0": 11.0,
            "seed": 420, "dof": "mid", "nod_phase": 2.1
        },
        # 14. Top Centre Crown Blossom (Stem 11: later bloomer, smaller)
        {
            "stem_idx": 11, "node_idx": -1, "offset": np.array([0.0, 0.05, 0.02]),
            "dir": np.array([0.04, 0.34, 0.94]), "scale": 0.92, "open": 0.86,
            "rot": 0.70, "ped_t0": 4.1, "petal_t0": 8.4, "stamen_t0": 11.6,
            "seed": 430, "dof": "mid", "nod_phase": 3.3
        },
        # 15. Front Low Centre-Right Cup (Stem 12: partially open, goblet-forward)
        {
            "stem_idx": 12, "node_idx": -1, "offset": np.array([0.0, 0.05, 0.04]),
            "dir": np.array([0.18, -0.02, 0.98]), "scale": 0.94, "open": 0.74,
            "rot": 0.20, "ped_t0": 4.2, "petal_t0": 8.2, "stamen_t0": 11.4,
            "seed": 440, "dof": "mid", "nod_phase": 4.4
        },
    ]
    # Stage 2 Closed Accent Buds (Stems 7, 8, 13)
    bud_heads = [
        {"stem_idx": 7, "t0": 3.4, "dof": "mid", "scale": 1.02},
        {"stem_idx": 8, "t0": 3.8, "dof": "bg",  "scale": 0.94},
        {"stem_idx": 13, "t0": 4.0, "dof": "mid", "scale": 0.98},
    ]
    # ---- Attachment planning (v2.4): every head gets an explicit receptacle point R and axis A.
    # Terminal heads: the stem is retargeted so stem tip + peduncle neck land exactly on the original head
    # centre R, and the neck ends coaxial with the flower axis A (no more arbitrary world-space offsets).
    def _stem_curve(s, n=64):
        uu = np.linspace(0.0, 1.0, n, dtype=np.float32)
        pp, tt = eval_cubic_bezier(s["p0"], s["p1"], s["p2"], s["p3"], uu)
        return pp, normalize_v3(tt)

    orig_curves = [_stem_curve(s) for s in stems_spec]
    bud_lean = {7: (0.16, 0.0, 0.22), 8: (-0.10, 0.0, 0.30), 13: (0.20, 0.0, 0.20)}
    heads = []
    for f in flowers_spec:
        h = dict(f)
        h["kind"] = "flower"
        pos0, tan0 = orig_curves[h["stem_idx"]]
        if h["node_idx"] == -1:
            h["R"] = (stems_spec[h["stem_idx"]]["p3"] + np.asarray(h["offset"])).astype(np.float32)
        else:
            h["R"] = (pos0[h["node_idx"]] + np.asarray(h["offset"])).astype(np.float32)
        h["A"] = normalize_v3(np.asarray(h["dir"], dtype=np.float32))
        h["chord"] = 0.19
        h["head_t0_min"] = h["ped_t0"] + 0.2
        heads.append(h)
    for bd in bud_heads:
        pos0, tan0 = orig_curves[bd["stem_idx"]]
        A = normalize_v3(tan0[-1] + np.array(bud_lean[bd["stem_idx"]], dtype=np.float32))
        h = dict(bd)
        h.update(kind="bud", node_idx=-1, A=A.astype(np.float32), open=0.0, rot=0.3 * bd["stem_idx"],
                 seed=250 + bd["stem_idx"], nod_phase=0.7 * bd["stem_idx"],
                 R=(pos0[-1] + 0.03 * A).astype(np.float32), chord=0.14, head_t0_min=bd["t0"] + 0.3)
        heads.append(h)
    for h in heads:
        if h["node_idx"] != -1:
            continue
        s = stems_spec[h["stem_idx"]]
        p2 = np.asarray(s["p2"], dtype=np.float32)
        t_est = normalize_v3(np.asarray(s["p3"], dtype=np.float32) - p2)
        p3n = h["R"] - h["chord"] * normalize_v3(t_est + h["A"])
        for _ in range(2):
            t_est = normalize_v3(p3n - p2)
            p3n = h["R"] - h["chord"] * normalize_v3(t_est + h["A"])
        s["p3"] = p3n.astype(np.float32)

    stem_data = []
    for s in stems_spec:
        el, pos_seq, tan_seq = generate_tulip_stem(
            p0=s["p0"], p1=s["p1"], p2=s["p2"], p3=s["p3"],
            r_base=s["r_base"], r_tip=s["r_tip"],
            Nu=64, Nv=23,
            wire_t0=s["wire_t0"], wire_t1=s["wire_t1"],
            surf_t0=s["surf_t0"], surf_t1=s["surf_t1"],
            rng_seed=s["seed"], dof_layer=s["dof"]
        )
        elements.append(el)
        stem_data.append((pos_seq, tan_seq, s))

    # 29 Broad Glaucous Leaves framing lower third and stem junctions
    leaves_spec = [
        # Lower basal shroud leaves wrapping bundle
        (0, 10, -0.85, 0.35, 1.25, 0.21, 0.5, "mid"),
        (0, 22,  0.88, 0.40, 1.15, 0.19, 0.9, "mid"),
        (1,  8, -1.20, 0.38, 1.28, 0.22, 0.6, "mid"),
        (1, 20,  0.72, 0.42, 1.12, 0.18, 1.0, "mid"),
        (2, 10,  1.15, 0.36, 1.20, 0.20, 0.7, "mid"),
        (2, 22, -0.65, 0.44, 1.10, 0.18, 1.2, "mid"),
        (3,  8, -1.30, 0.40, 1.18, 0.19, 0.8, "mid"),
        (3, 18,  0.60, 0.46, 1.05, 0.17, 1.3, "mid"),
        (4, 12,  0.95, 0.32, 1.15, 0.18, 1.0, "mid"),
        (4, 26, -0.82, 0.36, 1.02, 0.16, 1.5, "mid"),
        (5, 12, -0.95, 0.30, 1.10, 0.18, 1.1, "mid"),
        (5, 24,  0.78, 0.34, 0.98, 0.16, 1.6, "mid"),
        (6, 10, -1.35, 0.42, 1.22, 0.21, 1.1, "fg"),
        (6, 22,  0.55, 0.45, 1.05, 0.18, 1.5, "fg"),
        (7, 14,  0.80, 0.28, 0.95, 0.16, 1.4, "mid"),
        (7, 28, -0.70, 0.32, 0.88, 0.15, 1.8, "mid"),
        (8, 12,  0.50, 0.26, 0.98, 0.16, 1.5, "bg"),
        (8, 24, -0.52, 0.30, 0.90, 0.15, 1.9, "bg"),
        # Upper junction accent leaves
        (0, 36, -0.55, 0.45, 0.92, 0.16, 1.6, "mid"),
        (1, 34,  0.48, 0.48, 0.88, 0.15, 1.7, "mid"),
        (2, 34,  0.75, 0.42, 0.86, 0.15, 1.8, "mid"),
        (3, 30, -0.85, 0.50, 0.82, 0.14, 1.9, "mid"),
        # Leaves supporting the additional stems (v3)
        (9,  10, -1.10, 0.40, 1.12, 0.19, 1.2, "mid"),
        (9,  26,  0.70, 0.44, 0.92, 0.16, 1.7, "mid"),
        (10, 10,  1.05, 0.38, 1.15, 0.19, 1.3, "mid"),
        (10, 26, -0.60, 0.44, 0.92, 0.16, 1.8, "mid"),
        (11, 12,  0.85, 0.34, 0.95, 0.16, 1.6, "bg"),
        (12,  8, -0.80, 0.40, 1.00, 0.18, 1.8, "mid"),
        (12, 22,  0.75, 0.44, 0.88, 0.15, 2.0, "mid"),
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

    # ---- Heads: peduncle neck (with receptacle bulb) + 6 tepals (+3 petaloids) + androecium/pistil -------------
    for h in heads:
        pos_seq, tan_seq, s = stem_data[h["stem_idx"]]
        sc = h["scale"]
        dof = h["dof"]
        R, A = h["R"], h["A"]
        basis = build_basis_from_dir(A)
        terminal = (h["node_idx"] == -1)
        if terminal:
            q0, T0, r0 = pos_seq[-1], tan_seq[-1], s["r_tip"]
            neck_t0 = s["surf_t1"] - 0.10          # neck starts as the supporting stem finishes growing
        else:
            ni = h["node_idx"]
            q0, T0 = pos_seq[ni], tan_seq[ni]
            u_n = ni / float(len(pos_seq) - 1)
            r0 = 0.85 * (s["r_base"] * (1.0 - u_n) + s["r_tip"] * u_n)
            neck_t0 = s["surf_t1"] - 0.50
        neck_dur = 0.95
        q_end = R + A * (0.030 * sc)
        chord = float(np.linalg.norm(q_end - q0))
        if terminal:
            q1 = q0 + T0 * (0.42 * chord)
        else:
            q1 = q0 + normalize_v3(0.6 * T0 + 0.4 * normalize_v3(R - q0)) * (0.35 * chord)
        q2 = q_end - A * (0.42 * chord)
        el_neck, _, _ = generate_tulip_stem(
            p0=q0, p1=q1, p2=q2, p3=q_end,
            r_base=r0, r_tip=0.020 * sc, Nu=40, Nv=19,
            wire_t0=neck_t0 - 0.2, wire_t1=neck_t0 + neck_dur,
            surf_t0=neck_t0, surf_t1=neck_t0 + neck_dur,
            rng_seed=h["seed"] + 2, dof_layer=dof,
            cap_radius=0.036 * sc, cap_len=0.10 * sc
        )
        elements.append(el_neck)

        # the head only starts to appear once its neck has reached the receptacle
        head_t0 = max(h["head_t0_min"], neck_t0 + neck_dur - 0.10)
        tone = ((h["seed"] // 10) % 7 - 3) / 3.0 * 0.8
        base_rot = h["rot"]
        pt0 = h.get("petal_t0", 1.0e9)
        nod_phase = h["nod_phase"]
        is_bud = (h["kind"] == "bud")

        layers_spec = [("outer", base_rot + k * (2.0 * math.pi / 3.0), 0.20 * k, k) for k in range(3)]
        layers_spec += [("inner", base_rot + math.pi / 3.0 + k * (2.0 * math.pi / 3.0), 0.16 + 0.20 * k, 3 + k) for k in range(3)]
        if not is_bud:
            layers_spec += [("petaloid", base_rot + math.pi / 6.0 + k * (2.0 * math.pi / 3.0), 0.40 + 0.18 * k, 6 + k) for k in range(3)]
        for layer, az, p_delay, idx in layers_spec:
            elements.append(generate_tulip_tepal(
                center=R, basis=basis, azimuth=az, scale=sc, layer=layer,
                max_openness=h["open"],
                wire_t0=head_t0 - 0.15 + 0.05 * idx, wire_t1=head_t0 + 0.75 + 0.05 * idx,
                surf_t0=head_t0 + 0.05 * idx, surf_t1=head_t0 + 1.05 + 0.05 * idx,
                bloom_t0=pt0 + p_delay, bloom_t1=pt0 + p_delay + 2.4,
                rng_seed=h["seed"] + idx, dof_layer=dof, head_nod_phase=nod_phase, tone=tone
            ))

        if not is_bud:
            st0 = h["stamen_t0"]
            elements.extend(generate_tulip_androecium_and_pistil(
                center=R, basis=basis, scale=sc,
                wire_t0=st0 - 0.2, wire_t1=st0 + 1.0,
                surf_t0=st0, surf_t1=st0 + 1.5,
                bloom_t0=st0 + 0.1, bloom_t1=st0 + 2.0,
                rng_seed=h["seed"] + 50,
                dof_layer=dof, head_nod_phase=nod_phase
            ))

    return elements


# ============================================================================
# STUDIO BACKDROP, CAMERA PROJECTION & PBR/SSS SHADING
# ============================================================================
STUDIO_STYLE = "ivory"   # "ivory" | "pearl" | "champagne"

# (centre RGB, edge RGB, floor RGB) per premium light-studio treatment
_STUDIO_PALETTES = {
    "ivory":     ((250, 244, 232), (221, 207, 187), (206, 190, 168)),
    "pearl":     ((246, 245, 243), (214, 213, 214), (196, 195, 197)),
    "champagne": ((243, 230, 208), (206, 184, 156), (190, 166, 138)),
}


def create_tulip_studio_backdrop(style=None):
    """
    Premium light studio backdrop: soft warm key glow behind the bouquet, gentle vignette,
    slightly deeper floor falloff for studio depth. Light triangular dither avoids banding.
    """
    style = style or STUDIO_STYLE
    c_ctr, c_edge, c_floor = [np.array(c, dtype=np.float32) for c in _STUDIO_PALETTES[style]]
    y = np.linspace(-1.0, 1.0, HEIGHT, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32)[None, :]

    r2_center = (x * 0.80) ** 2 + ((y + 0.08) * 0.78) ** 2
    glow = np.exp(-1.25 * r2_center)[..., None]

    # Warm, slightly off-axis key bloom (upper-left behind the subject)
    r2_key = ((x + 0.18) * 0.95) ** 2 + ((y + 0.30) * 0.90) ** 2
    key = np.exp(-2.4 * r2_key)[..., None]

    # Floor falloff: gently deeper toward the bottom of frame
    floor = smoothstep(0.10, 1.25, y)[..., None]

    rgb = c_edge[None, None, :] * (1.0 - glow) + c_ctr[None, None, :] * glow
    rgb = rgb + 5.0 * key
    rgb = rgb * (1.0 - 0.38 * floor) + c_floor[None, None, :] * (0.38 * floor) * (0.65 + 0.35 * glow)

    rng = np.random.RandomState(42)
    dither = (rng.uniform(-0.5, 0.5, size=rgb.shape) + rng.uniform(-0.5, 0.5, size=rgb.shape)).astype(np.float32)
    rgb = np.clip(rgb + dither, 0.0, 255.0)
    return np.rint(rgb[..., ::-1]).astype(np.uint8)   # RGB -> BGR


STUDIO_BG = create_tulip_studio_backdrop()


def project_points(P_world, cam_rot, cam_target, cam_dist, fov_scale):
    P_rel = (P_world - cam_target) @ cam_rot.T
    z_depth = cam_dist - P_rel[..., 2]
    z_safe = np.maximum(z_depth, 0.25)
    sx = (WIDTH * 0.50) + (P_rel[..., 0] / z_safe) * fov_scale
    sy = (HEIGHT * 0.52) - (P_rel[..., 1] / z_safe) * fov_scale
    return sx, sy, z_depth, P_rel


def compute_lighting(normals_cam, base_rgb, sss_strength, spec_strength, shininess, b_prog=1.0, u_quad=None, sss_tint=None):
    """
    Physically-based 3D botanical shader:
      - Neutral-warm key light from upper-left-front
      - Soft cool sky fill light from the right
      - Rim backlight + Fresnel + Subsurface Scattering (SSS) for translucent tepals
      - Micro-cuticle Blinn-Phong specular highlight
      - Gentle filmic shoulder that preserves delicate colour gradients
    `b_prog` and `u_quad` are accepted for API compatibility.
    """
    L_key = normalize_v3(np.array([-0.38, 0.58, 0.72], dtype=np.float32))
    L_fill = normalize_v3(np.array([0.68, -0.18, 0.71], dtype=np.float32))
    L_rim = normalize_v3(np.array([0.15, 0.75, -0.64], dtype=np.float32))
    L_bounce = normalize_v3(np.array([0.10, -0.80, 0.55], dtype=np.float32))
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
    # Satin/waxy cuticle: tight highlight + broad soft sheen lobe
    spec = spec_strength * (ndoth ** shininess) + 0.45 * spec_strength * (ndoth ** 7.0)

    # Warm ivory bounce from the studio floor/walls (soft fill from below)
    ndotl_b = np.sum(N * L_bounce, axis=-1, keepdims=True)
    diff_bounce = np.clip((ndotl_b + 0.25) / 1.25, 0.0, 1.0)
    bounce_col = np.array([1.00, 0.90, 0.78], dtype=np.float32)

    key_col = np.array([1.02, 0.98, 0.95], dtype=np.float32)
    fill_col = np.array([0.78, 0.82, 0.92], dtype=np.float32)
    sss_col = np.array([0.74, 0.86, 1.04], dtype=np.float32) if sss_tint is None else sss_tint

    lit_rgb = (
        base_rgb * (0.24 + 0.66 * diff_key * key_col + 0.20 * diff_fill * fill_col + 0.12 * diff_bounce * bounce_col)
        + base_rgb * sss * sss_col
        + spec * np.array([1.0, 0.97, 0.95], dtype=np.float32)
    )

    lit_rgb = np.clip(lit_rgb, 0.0, 1.25)
    lit_rgb = lit_rgb / (1.0 + 0.20 * lit_rgb) * 1.12
    return np.clip(lit_rgb, 0.0, 1.0)


# ============================================================================
# SINGLE-FRAME 3D RENDERER (WITH DEPTH OF FIELD & BOKEH)
# ============================================================================
PETAL_SSS_TINT = np.array([0.30, 0.62, 1.10], dtype=np.float32)   # translucent edges glow azure, not white
SCENE_ELEMENTS = None


PFLOWERS = []
POLLEN_PER_FLOWER = 16


def build_pollen_registry(elements):
    """Group golden anthers by flower so each open tulip can emit its own pollen shimmer."""
    groups = {}
    for el in elements:
        if el.pollen_tip is not None and el.anchor is not None:
            groups.setdefault(tuple(np.round(el.anchor, 5)), []).append(el)
    reg = []
    for key, els in groups.items():
        anchor = np.array(key, dtype=np.float32)
        opens = np.array([e.pollen_tip[1] for e in els], dtype=np.float32)
        mean_open = opens.mean(axis=0)
        axis = mean_open - anchor
        unit = float(np.linalg.norm(axis))
        if unit < 1e-6:
            continue
        axis = axis / unit
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float32) if abs(axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        p1 = normalize_v3(np.cross(axis, helper))
        p2 = normalize_v3(np.cross(axis, p1))
        reg.append({
            "anchor": anchor, "els": els, "axis": axis, "p1": p1, "p2": p2, "unit": unit,
            "t_on": min(e.surf_t0 for e in els) + 1.1, "dof": els[0].dof_layer,
        })
    return reg


def init_worker(w=None, h=None, show_hud=False, style=None):
    global SCENE_ELEMENTS, WIDTH, HEIGHT, STUDIO_BG, SHOW_HUD, STUDIO_STYLE, PFLOWERS
    if style is not None:
        STUDIO_STYLE = style
    if w is not None and h is not None:
        WIDTH, HEIGHT = w, h
    STUDIO_BG = create_tulip_studio_backdrop()
    SHOW_HUD = show_hud
    SCENE_ELEMENTS = build_tulip_scene()
    PFLOWERS = build_pollen_registry(SCENE_ELEMENTS)


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
    global PFLOWERS
    if SCENE_ELEMENTS is None:
        SCENE_ELEMENTS = build_tulip_scene()
        PFLOWERS = build_pollen_registry(SCENE_ELEMENTS)

    t = (frame_idx / max(1, total_frames - 1)) * DURATION
    p_all = frame_idx / max(1, total_frames - 1)

    bg_fade = 0.35 + 0.65 * smootherstep(0.0, 1.5, t)
    canvas = (STUDIO_BG.astype(np.float32) * (0.90 + 0.10 * smootherstep(0.0, 1.5, t))).astype(np.uint8)
    bg_only = canvas.copy()

    # 1. Camera Trajectory
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
            P_s, normals_s, c_quad_s, u_quad = slice_mesh_along_u(P_full, C_full, s_prog)
            normals_cam = normals_s @ cam_rot.T
            sx, sy, zd, _ = project_points(P_s, cam_rot, cam_target, cam_dist, fov_scale)

            if c_quad_s.shape[-1] == 6:
                # two-sided petal pigment: outer-surface look vs inner-surface look, chosen by facing
                c_quad_s = np.where(normals_cam[..., 2:3] >= 0.0, c_quad_s[..., :3], c_quad_s[..., 3:])

            lit_rgb = compute_lighting(
                normals_cam, c_quad_s, el.sss_strength, el.spec_strength, el.shininess,
                b_prog=b_prog, u_quad=u_quad,
                sss_tint=PETAL_SSS_TINT if el.kind == "petal" else None
            )

            # Growth Wavefront: rim_glow & wave_particles on leading edge
            if s_prog < 0.992:
                dist_edge = s_prog - u_quad
                rim_glow = np.exp(-((dist_edge / 0.075) ** 2)) * (dist_edge >= -0.02)
                glow_col = np.array([0.90, 0.90, 0.92], dtype=np.float32)
                lit_rgb = np.clip(lit_rgb + 0.45 * rim_glow[..., None] * glow_col, 0.0, 1.0)

                if s_prog > 0.04:
                    tip_row = P_s.shape[0] - 1
                    for j_idx in range(0, P_s.shape[1], 4):
                        wave_particles.append((
                            float(sx[tip_row, j_idx]),
                            float(sy[tip_row, j_idx]),
                            (185, 215, 255),
                            2.2
                        ))

            # Seam-Free Quads: 0.55px Outward Quad Expansion
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
            z_q = (0.25 * (zd[:-1, :-1] + zd[1:, :-1] + zd[1:, 1:] + zd[:-1, 1:])).reshape(-1)
            bgr_q = np.rint(lit_rgb[..., ::-1] * 255.0).astype(np.uint8).reshape(-1, 3)

            bucket = layers[el.dof_layer]
            bucket["pts"].append(pts_int)
            bucket["z"].append(z_q)
            bucket["bgr"].append(bgr_q)

            # Anther Pollen Beads
            if el.pollen_tip is not None and s_prog > 0.65:
                pos_c, pos_o, rad_px, core_bgr, rim_bgr = el.pollen_tip
                tip_3d = pos_c * (1.0 - b_prog) + pos_o * b_prog
                if wind_vec is not None and el.anchor is not None:
                    dist_a = float(np.linalg.norm(tip_3d - el.anchor))
                    tip_3d = tip_3d + (dist_a ** 1.35) * 0.22 * wind_vec
                tsx, tsy, tzd, _ = project_points(tip_3d[None, :], cam_rot, cam_target, cam_dist, fov_scale)
                anther_scale = smootherstep(0.85, 1.0, s_prog)
                r_int = max(2, int(round(rad_px * anther_scale)))
                bucket["pollen"].append((
                    float(tzd[0]) - 0.012,
                    int(round(float(tsx[0]))),
                    int(round(float(tsy[0]))),
                    r_int, core_bgr, rim_bgr
                ))

        # Wireframe Splines
        base_wire_vis = 1.0 - 0.88 * smootherstep(0.55, 1.0, s_prog)
        if el.extra_wire_t0 is not None and el.extra_wire_t1 is not None:
            unfold_pulse = math.sin(smootherstep(el.extra_wire_t0, el.extra_wire_t1, t) * math.pi)
            base_wire_vis = max(base_wire_vis, 0.58 * unfold_pulse)

        wire_vis = global_wire_alpha * base_wire_vis
        if w_prog > 0.01 and wire_vis > 0.04:
            P_w, _, _, _ = slice_mesh_along_u(P_full, C_full, w_prog)
            wsx, wsy, wzd, _ = project_points(P_w, cam_rot, cam_target, cam_dist, fov_scale)
            Nu_w, Nv_w = P_w.shape[:2]
            # Continuous tail fade: avoids a one-frame wireframe pop when wire_vis crosses the cutoff
            wire_fade = float(smoothstep(0.04, 0.10, wire_vis))
            wc_bgr = tuple(int(round(c * 255.0 * (0.30 + 0.70 * wire_vis) * wire_fade)) for c in el.wire_color[::-1])

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

    # 2b. Golden pollen shimmer: tiny champagne-gold motes rising from each open flower's anthers
    pollen_glow = []
    for pf in PFLOWERS:
        t_on = pf["t_on"]
        intro = float(smootherstep(t_on, t_on + 2.6, t))
        if intro <= 0.01:
            continue
        unit, axis, pp1, pp2 = pf["unit"], pf["axis"], pf["p1"], pf["p2"]
        bucket = layers[pf["dof"]]
        els = pf["els"]
        anchor = pf["anchor"]

        # Warm illuminated centre
        ctr3 = anchor + axis * (unit * 0.85)
        csx, csy, czd, _ = project_points(ctr3[None, :], cam_rot, cam_target, cam_dist, fov_scale)
        breathe = 0.80 + 0.20 * math.sin(t * 1.7 + float(anchor[0]) * 9.0)
        pollen_glow.append((float(csx[0]), float(csy[0]), 0.95 * unit * fov_scale / max(float(czd[0]), 0.5), 0.14 * intro * breathe))

        for j in range(POLLEN_PER_FLOWER):
            el = els[j % len(els)]
            ph = (j * 0.6180339 + 0.13 * (j % 3)) % 1.0
            period = 3.4 + 1.8 * (((j * 7) % 5) / 4.0)
            life = ((t - t_on) / period + ph) % 1.0
            pos_c, pos_o = el.pollen_tip[0], el.pollen_tip[1]
            b_prog = float(smootherstep(el.bloom_t0, el.bloom_t1, t))
            a_pos = pos_c * (1.0 - b_prog) + pos_o * b_prog
            dist_a = float(np.linalg.norm(a_pos - anchor))
            a_pos = a_pos + (dist_a ** 1.35) * 0.22 * wind_vec
            ang = 2.0 * math.pi * (life * 1.3 + ph * 3.0)
            pos = (a_pos
                   + axis * (1.15 * unit * life)
                   + pp1 * (0.35 * unit * math.sin(ang) * (0.35 + life))
                   + pp2 * (0.35 * unit * math.cos(ang * 0.8) * (0.35 + life)))
            twinkle = 0.55 + 0.45 * math.sin(t * 5.2 + ph * 23.0)
            alpha = (math.sin(math.pi * life) ** 1.3) * twinkle * intro
            if alpha < 0.14:
                continue
            psx, psy, pzd, _ = project_points(pos[None, :], cam_rot, cam_target, cam_dist, fov_scale)
            ix, iy = int(round(float(psx[0]))), int(round(float(psy[0])))
            r_px = 1 if (alpha < 0.65 or j % 5) else 2
            core = (int(70 + 120 * alpha), int(195 + 43 * alpha), int(238 + 14 * alpha))   # BGR champagne-gold
            rim = (int(60 + 40 * alpha), int(150 + 55 * alpha), int(205 + 35 * alpha))
            bucket["pollen"].append((float(pzd[0]) - 0.02, ix, iy, r_px, core, rim))
            pollen_glow.append((float(psx[0]), float(psy[0]), 3.5 + 2.5 * alpha, 0.55 * alpha))

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
        c_quad_d = np.where(normals_cam_d[..., 2:3] >= 0.0, c_quad_d[..., :3], c_quad_d[..., 3:])
        sx_d, sy_d, zd_d, _ = project_points(P_d, cam_rot, cam_target, cam_dist, fov_scale)

        lit_d = compute_lighting(normals_cam_d, c_quad_d, 0.34, 0.17, 26.0, b_prog=1.0, sss_tint=PETAL_SSS_TINT)
        q_x = np.stack([sx_d[:-1, :-1], sx_d[1:, :-1], sx_d[1:, 1:], sx_d[:-1, 1:]], axis=-1)
        q_y = np.stack([sy_d[:-1, :-1], sy_d[1:, :-1], sy_d[1:, 1:], sy_d[:-1, 1:]], axis=-1)
        pts_int = np.rint(np.stack([q_x, q_y], axis=-1).reshape(-1, 4, 2) * SUBPIX).astype(np.int32)
        z_q = (0.25 * (zd_d[:-1, :-1] + zd_d[1:, :-1] + zd_d[1:, 1:] + zd_d[:-1, 1:])).reshape(-1)
        bgr_q = np.rint(lit_d[..., ::-1] * 255.0).astype(np.uint8).reshape(-1, 3)

        bucket = layers[d["dof"]]
        bucket["pts"].append(pts_int)
        bucket["z"].append(z_q)
        bucket["bgr"].append(bgr_q)

    # 4. Composite Depth-of-Field Layers (bg sigma=3.2, fg sigma=2.4)
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

    # 4b. Soft studio shadow cast by the bouquet onto the backdrop (warm taupe, double-lobed)
    diff_m = np.abs(canvas.astype(np.int16) - bg_only.astype(np.int16)).max(axis=2)
    subj = (diff_m > 5).astype(np.float32)
    subj_f = cv2.GaussianBlur(subj, (0, 0), 0.8)
    hs = HEIGHT / 1080.0
    far = cv2.warpAffine(subj, np.float32([[1, 0, 24 * hs], [0, 1, 34 * hs]]), (WIDTH, HEIGHT))
    near = cv2.warpAffine(subj, np.float32([[1, 0, 9 * hs], [0, 1, 13 * hs]]), (WIDTH, HEIGHT))
    shade = 0.24 * cv2.GaussianBlur(far, (0, 0), 28 * hs) + 0.14 * cv2.GaussianBlur(near, (0, 0), 7 * hs)
    shade = np.clip(shade, 0.0, 0.45)[..., None] * (1.0 - subj_f[..., None])
    canvas = np.clip(canvas.astype(np.float32) * (1.0 - shade * np.array([0.95, 0.80, 0.62], dtype=np.float32)), 0, 255).astype(np.uint8)

    # 5. Wireframe Overlay (deep-blue construction lines on the light studio)
    if wire_lines:
        wire_layer = np.zeros_like(canvas)
        wire_lines.sort(key=lambda item: item[1], reverse=True)
        for pts_sub, _, wc_bgr, thick in wire_lines:
            cv2.polylines(
                wire_layer, [pts_sub], isClosed=False, color=wc_bgr,
                thickness=thick, lineType=cv2.LINE_AA, shift=SHIFT
            )
        wl = wire_layer.astype(np.float32) / 255.0
        wm = wl.max(axis=2, keepdims=True)
        hue = wl / np.maximum(wm, 1e-4)
        wa = np.clip(wm * 1.6, 0.0, 1.0) * 0.30
        canvas = np.clip(canvas.astype(np.float32) * (1.0 - wa) + (hue * 0.72 * 255.0) * wa, 0, 255).astype(np.uint8)

    # 6. Atmospheric dust, growth sparkles and faint bokeh (alpha-composited so they read on a light studio)
    H_, W_ = canvas.shape[:2]
    dust_a = np.zeros((H_, W_), dtype=np.float32)
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
            twinkle = max(0.0, (0.35 + 0.65 * math.sin(t * 2.4 + amb_phase[i]))) * bg_fade
            rad = int(round((1.5 + 2.8 * (i % 3)) * (4.2 / max(2.2, azd[i]))))
            cv2.circle(dust_a, (px, py), max(1, rad), float(min(1.0, twinkle)), -1, lineType=cv2.LINE_AA)

    for px, py, bgr, rad in wave_particles[:240]:
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < WIDTH and 0 <= iy < HEIGHT:
            cv2.circle(dust_a, (ix, iy), max(1, int(round(rad))), 0.95, -1, lineType=cv2.LINE_AA)

    dust_soft = 0.85 * cv2.GaussianBlur(dust_a, (0, 0), 1.4) + 0.55 * cv2.GaussianBlur(dust_a, (0, 0), 3.5)
    dust_alpha = np.clip(dust_soft * 0.70, 0.0, 0.80)[..., None]
    dust_col = np.array([112, 178, 226], dtype=np.float32)   # deep champagne-gold (BGR)
    canvas = np.clip(canvas.astype(np.float32) * (1.0 - dust_alpha) + dust_col * dust_alpha, 0, 255).astype(np.uint8)

    # Delicate out-of-focus bokeh discs (a few, very faint) for depth separation
    rng_b = np.random.RandomState(211)
    n_bokeh = 12
    bk_pos = rng_b.uniform([-1.4, -1.9, -1.2], [1.4, 1.9, 0.9], size=(n_bokeh, 3)).astype(np.float32)
    bk_phase = rng_b.uniform(0.0, 2.0 * math.pi, size=n_bokeh)
    bk_pos[:, 1] += 0.08 * t + 0.05 * np.sin(t * 0.9 + bk_phase)
    bk_pos[:, 1] = ((bk_pos[:, 1] + 1.9) % 3.8) - 1.9
    bsx, bsy, bzd, _ = project_points(bk_pos, cam_rot, cam_target, cam_dist, fov_scale)
    bk_gold = np.zeros((H_, W_), dtype=np.float32)
    bk_white = np.zeros((H_, W_), dtype=np.float32)
    for i in range(n_bokeh):
        bx, by = int(round(bsx[i])), int(round(bsy[i]))
        if 0 <= bx < WIDTH and 0 <= by < HEIGHT:
            twinkle = (0.50 + 0.50 * math.sin(t * 1.6 + bk_phase[i])) * bg_fade
            r_orb = int(round((10.0 + (i % 4) * 4.0) * (HEIGHT / 1080.0)))
            cv2.circle(bk_gold if i % 4 == 0 else bk_white, (bx, by), r_orb, float(0.30 * twinkle), -1, lineType=cv2.LINE_AA)
    bk_gold = cv2.GaussianBlur(bk_gold, (0, 0), 6.0)[..., None]
    bk_white = cv2.GaussianBlur(bk_white, (0, 0), 6.0)[..., None]
    cf = canvas.astype(np.float32)
    cf = cf * (1.0 - bk_gold) + np.array([150, 205, 236], dtype=np.float32) * bk_gold
    cf = cf * (1.0 - bk_white) + np.array([255, 252, 248], dtype=np.float32) * bk_white
    canvas = np.clip(cf, 0, 255).astype(np.uint8)

    # Pollen glow: soft additive champagne light concentrated at the flower centres
    if pollen_glow:
        pg = np.zeros_like(canvas)
        for gx, gy, gr, ga in pollen_glow:
            ix, iy = int(round(gx)), int(round(gy))
            if 0 <= ix < WIDTH and 0 <= iy < HEIGHT:
                c = (int(95 * ga), int(205 * ga), int(255 * ga))
                cv2.circle(pg, (ix, iy), max(1, int(round(gr))), c, -1, lineType=cv2.LINE_AA)
        pg_soft = cv2.GaussianBlur(pg, (0, 0), 3.2)
        pg_wide = cv2.GaussianBlur(pg, (0, 0), 9.0)
        canvas = cv2.addWeighted(canvas, 1.0, pg_soft, 0.75, 0.0)
        canvas = cv2.addWeighted(canvas, 1.0, pg_wide, 0.55, 0.0)

    # 7. Subtle optical highlight bloom (true highlights only; background excluded)
    subj_halo = cv2.GaussianBlur(subj, (0, 0), 10.0)[..., None]
    bright = np.maximum(canvas.astype(np.float32) - 226.0, 0.0) * np.clip(subj_halo * 1.5, 0.0, 1.0)
    bloom = cv2.GaussianBlur(bright, (0, 0), sigmaX=14.0)
    canvas = np.clip(canvas.astype(np.float32) + 0.30 * bloom, 0, 255).astype(np.uint8)

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
        overlay, "3D PROCEDURAL BOTANICAL SPECIMEN  //  9-TEPAL LAYERED ANTHESIS", (68, 152),
        cv2.FONT_HERSHEY_SIMPLEX, 0.41, c_muted, 1, cv2.LINE_AA
    )
    cv2.line(overlay, (68, 172), (450, 172), (85, 105, 125), 1, cv2.LINE_AA)

    if t < 3.0:
        stage_str = "STAGE 01 // CAULESCENT STEM & GLAUCOUS LEAF GROWTH"
        eq_str = "B(u,v) : Cubic Bezier Parallel Transport + Lanceolate Foliage"
    elif t < 6.0:
        stage_str = "STAGE 02 // IMBRICATE 6-TEPAL BUD & COAXIAL NECK"
        eq_str = "H(u) : Closed 6-Tepal Imbrication + Receptacle Dome"
    elif t < 9.0:
        stage_str = "STAGE 03 // 9-TEPAL GOBLET UNFOLDING & AZURE SSS"
        eq_str = "P(u,v) : Pre-solved P_states + Two-Sided Sapphire Pigment"
    elif t < 13.0:
        stage_str = "STAGE 04 // 6 GOLDEN STAMENS, 3-LOBED PISTIL & DEHISCENCE"
        eq_str = "Bilobed Anther Capsules + Golden Filaments + Pollen Shedding"
    else:
        stage_str = "STAGE 05 // CINEMATIC SPECIMEN & DRIFTING SAPPHIRE TEPALS"
        eq_str = "3D Depth-of-Field Orbit + Tumbling Tepal Aerodynamics"

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
    cv2.rectangle(overlay, (68, y_bot + 46), (68 + bar_w, y_bot + 50), (45, 50, 68), -1)
    cv2.rectangle(
        overlay, (68, y_bot + 46), (68 + int(bar_w * prog), y_bot + 50),
        (235, 195, 135), -1
    )

    cv2.addWeighted(overlay, hud_alpha * 0.82, canvas, 1.0 - hud_alpha * 0.82, 0.0, dst=canvas)


# ============================================================================
# CLI ENTRY POINT (PREVIEW KEYFRAMES OR FULL MP4 RENDER)
# ============================================================================
def render_preview_keyframes(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    init_worker(WIDTH, HEIGHT, SHOW_HUD, STUDIO_STYLE)
    test_times = [6.0, 9.0, 11.0, 13.0, 17.8]
    aspect_tag = f"{WIDTH}x{HEIGHT}"
    for sec in test_times:
        f_idx = int(round((sec / DURATION) * (TOTAL_FRAMES - 1)))
        t0 = time.time()
        img = render_frame(f_idx, TOTAL_FRAMES)
        out_path = os.path.join(out_dir, f"preview_{aspect_tag}_{sec:.1f}s.png")
        cv2.imwrite(out_path, img)
        print(f"Rendered {out_path} in {time.time() - t0:.2f}s", flush=True)


def find_ffmpeg_binary():
    local_binary = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
    if os.path.isfile(local_binary):
        return local_binary
    which_bin = shutil.which("ffmpeg")
    if which_bin:
        return which_bin
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return None


def render_full_video(output_mp4, max_frames=None):
    ffmpeg_bin = find_ffmpeg_binary()
    target_frames = TOTAL_FRAMES if max_frames is None else min(TOTAL_FRAMES, max_frames)

    print(f"Rendering {target_frames}/{TOTAL_FRAMES} frames at {WIDTH}x{HEIGHT} ({FPS} fps, {DURATION:.1f}s)...", flush=True)
    t_start = time.time()

    proc = None
    cv_writer = None

    if ffmpeg_bin:
        print(f"Using FFmpeg binary: {ffmpeg_bin}", flush=True)
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{WIDTH}x{HEIGHT}",
            "-pix_fmt", "bgr24",
            "-r", str(FPS),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-g", "1",
            "-bf", "0",
            "-movflags", "+faststart",
            output_mp4
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        print("FFmpeg not found; using OpenCV VideoWriter fallback", flush=True)
        cv_writer = cv2.VideoWriter(
            output_mp4,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (WIDTH, HEIGHT)
        )
        if not cv_writer.isOpened():
            raise RuntimeError(f"Failed to initialize OpenCV VideoWriter for {output_mp4}")

    num_workers = min(8, os.cpu_count() or 4)
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(WIDTH, HEIGHT, SHOW_HUD, STUDIO_STYLE)) as pool:
        for idx, frame in enumerate(pool.imap(render_frame, range(target_frames), chunksize=4)):
            if proc is not None:
                proc.stdin.write(frame.tobytes())
            else:
                cv_writer.write(frame)

            if (idx + 1) % 10 == 0 or (idx + 1) == target_frames:
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
    parser = argparse.ArgumentParser(description="Procedural 3D Blue Tulip Botanical Animation Renderer (v2.4.0)")
    parser.add_argument("--aspect", choices=["portrait", "landscape"], default="landscape",
                        help="Render aspect ratio: landscape (1920x1080) or portrait (1080x1920)")
    parser.add_argument("--preview", action="store_true", help="Render preview keyframe images to ./previews/")
    parser.add_argument("--frames", type=int, default=None, help="Render only first N frames")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 filename")
    parser.add_argument("--hud", action="store_true", help="Draw botanical specimen telemetry overlay")
    parser.add_argument("--backdrop", choices=["ivory", "pearl", "champagne"], default="ivory",
                        help="Light studio backdrop treatment (default: ivory)")
    args = parser.parse_args()

    if args.aspect == "portrait":
        WIDTH, HEIGHT = 1080, 1920
    else:
        WIDTH, HEIGHT = 1920, 1080

    SHOW_HUD = args.hud
    STUDIO_STYLE = args.backdrop

    if args.preview:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "previews")
        render_preview_keyframes(out_dir)
    else:
        out_name = args.out or (f"tulip_{args.aspect}.mp4")
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_name)
        render_full_video(out_path, max_frames=args.frames)
