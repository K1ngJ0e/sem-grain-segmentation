"""Automatic semantic thresholding and morphology-based instance separation.

Adapted without numerical changes from the author-supplied pipeline.
The empty-high-seed hysteresis fallback returns the low-threshold mask.
"""

import cv2
import numpy as np


def binarize_single(prob, th=0.5):
    return (prob >= th).astype(np.uint8)

def binarize_hysteresis(prob, low=0.35, high=0.60):
    low_mask  = (prob >= low).astype(np.uint8)
    high_mask = (prob >= high).astype(np.uint8)
    if high_mask.sum() == 0:
        return low_mask
    num, labels = cv2.connectedComponents(low_mask, connectivity=8)
    out = np.zeros_like(low_mask)
    for i in range(1, num):
        comp = (labels == i)
        if (high_mask[comp] == 1).any():
            out[comp] = 1
    return out


# -----------------------------
# 10) Instance split: Hybrid Morph (boundary-cut + local watershed)
# -----------------------------
def boundary_prob_from_fgprob(fg_prob):
    p = fg_prob.astype(np.float32)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = grad / (grad.max() + 1e-6)
    return np.clip(grad, 0.0, 1.0)

def nms_peaks_from_dist(dist_n, min_dist=6, peak_th=0.35):
    d = dist_n.astype(np.float32)
    r = max(int(min_dist), 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    dil = cv2.dilate(d, ker, iterations=1)
    peaks = ((d >= dil - 1e-6) & (d >= float(peak_th))).astype(np.uint8)
    peaks = cv2.morphologyEx(
        peaks, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1
    )
    return peaks

def remove_small_and_relabel(label_map, min_area=60):
    if label_map.max() <= 0:
        return label_map.astype(np.int32), 0
    out = np.zeros_like(label_map, dtype=np.int32)
    k = 0
    for i in range(1, int(label_map.max()) + 1):
        area = int((label_map == i).sum())
        if area >= int(min_area):
            k += 1
            out[label_map == i] = k
    return out, k

def split_instances_hybrid_morph(
    fg_prob, fg_mask01,
    bnd_th=0.25, bnd_close=3, bnd_dilate=1,
    big_area=1200,
    seed_erode=1, seed_min_dist=5, seed_peak_th=0.32,
    w_dist=0.92, w_bnd=0.08,
    min_instance_area=60
):
    fg = fg_mask01.astype(np.uint8)

    # boundary mask: |∇p| -> close -> dilate
    bprob = boundary_prob_from_fgprob(fg_prob)
    bmask = (bprob >= float(bnd_th)).astype(np.uint8)

    if bnd_close > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*bnd_close+1, 2*bnd_close+1))
        bmask = cv2.morphologyEx(bmask, cv2.MORPH_CLOSE, ker, iterations=1)

    if bnd_dilate > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*bnd_dilate+1, 2*bnd_dilate+1))
        bmask = cv2.dilate(bmask, ker, iterations=1)

    # cut fg by boundary
    cut = fg.copy()
    cut[bmask == 1] = 0

    # CC on cut
    num, lab0 = cv2.connectedComponents(cut, connectivity=8)

    out = np.zeros_like(lab0, dtype=np.int32)
    next_id = 0

    # local watershed for large blobs
    for cid in range(1, num):
        comp = (lab0 == cid).astype(np.uint8)
        area = int(comp.sum())
        if area == 0:
            continue

        if area < int(big_area):
            next_id += 1
            out[comp == 1] = next_id
            continue

        comp_for_dist = comp.copy()
        if seed_erode > 0:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*seed_erode+1, 2*seed_erode+1))
            comp_for_dist = cv2.erode(comp_for_dist, ker, iterations=1)
            if int(comp_for_dist.sum()) == 0:
                comp_for_dist = comp

        dist = cv2.distanceTransform(comp_for_dist, cv2.DIST_L2, 5)
        dist_n = dist / (dist.max() + 1e-6)

        peaks = nms_peaks_from_dist(dist_n, min_dist=seed_min_dist, peak_th=seed_peak_th)
        peaks[comp == 0] = 0
        nseed, seed_lab = cv2.connectedComponents(peaks, connectivity=8)

        # if not enough seeds -> keep as one instance
        if nseed <= 2:
            next_id += 1
            out[comp == 1] = next_id
            continue

        # height map: mostly (1-dist) + small boundary guidance
        height = w_dist * (1.0 - dist_n) + w_bnd * bprob
        height = np.clip(height, 0.0, 1.0)
        height_u8 = (255.0 * height).astype(np.uint8)
        ws_bgr = cv2.cvtColor(height_u8, cv2.COLOR_GRAY2BGR)

        markers = np.zeros_like(comp, dtype=np.int32)
        markers[comp == 0] = 1
        markers[seed_lab > 0] = seed_lab[seed_lab > 0] + 1

        mk_ws = cv2.watershed(ws_bgr, markers)

        inst = mk_ws.copy()
        inst[inst <= 1] = 0
        inst[inst > 1] -= 1
        inst[comp == 0] = 0

        for si in range(1, int(inst.max()) + 1):
            part = (inst == si)
            if int(part.sum()) == 0:
                continue
            next_id += 1
            out[part] = next_id

    out, k = remove_small_and_relabel(out, min_area=min_instance_area)
    debug = {"bprob": bprob, "bmask": bmask, "cut": cut}
    return out, k, debug


# -----------------------------
# 11) Visualization helpers
# -----------------------------
def overlay_gray(img_u8, mask01, color=(255,0,0)):
    vis = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    m = mask01.astype(bool)
    vis[m] = (0.5*vis[m] + 0.5*np.array(color)).astype(np.uint8)
    return vis

def colorize_labels(label_map, num_labels):
    h, w = label_map.shape
    out = np.zeros((h,w,3), np.uint8)
    rng = np.random.RandomState(123)
    colors = rng.randint(0, 255, size=(max(num_labels,2), 3), dtype=np.uint8)
    colors[0] = 0
    for i in range(1, num_labels):
        out[label_map==i] = colors[i]
    return out

def overlay_instance_boundaries_morph(img_u8: np.ndarray, label_map: np.ndarray, thickness: int = 1):
    """
    干净边界：mask - erode(mask)（不依赖 Sobel）
    """
    vis = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    if label_map is None or int(label_map.max()) <= 0:
        return vis

    lab = label_map.astype(np.int32)
    edge = np.zeros(lab.shape, np.uint8)

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    iters = max(int(thickness), 1)

    for i in range(1, int(lab.max()) + 1):
        m = (lab == i).astype(np.uint8)
        if int(m.sum()) == 0:
            continue
        er = cv2.erode(m, ker, iterations=iters)
        b = (m - er) > 0
        edge[b] = 1

    vis[edge == 1] = (255, 255, 255)
    return vis

def overlay_instances_alpha(img_u8: np.ndarray, label_map: np.ndarray, alpha: float = 0.45, seed: int = 123):
    """
    instance 彩色区域 alpha 叠加到原图
    """
    base = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    if label_map is None or int(label_map.max()) <= 0:
        return base

    lab = label_map.astype(np.int32)
    H, W = lab.shape
    K = int(lab.max())

    rng = np.random.RandomState(seed)
    colors = rng.randint(0, 255, size=(K + 1, 3), dtype=np.uint8)
    colors[0] = 0

    color_img = np.zeros((H, W, 3), np.uint8)
    for i in range(1, K + 1):
        color_img[lab == i] = colors[i]

    m = lab > 0
    out = base.copy()
    out[m] = (alpha * color_img[m] + (1 - alpha) * out[m]).astype(np.uint8)
    return out
