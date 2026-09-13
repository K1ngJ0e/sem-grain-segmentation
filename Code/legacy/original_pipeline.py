# ==========================================================
# FULL PIPELINE (DIRECTLY RUNNABLE) — Hybrid Morphology Optimization
# + Manual pred-prob fix (file / OpenCV GUI / Matplotlib interactive) ✅
# + Instance overlay visualization ✅
#
# 1) Train: Unet++ + ResNet18 + Encoder-SE + Tversky
# 2) Mask reading: RED=FG, BLACK=BG (robust HSV)
# 3) Visuals: raw/GT/prob/pred/overlays + debug (boundary/cut)
# 4) Threshold ablation: single vs hysteresis (FG)
# 5) Instance split (STRONG + MORPH):
#    - boundary from |∇p| (fg_prob gradient)
#    - MORPH CLOSE to connect broken boundaries + DILATE to thicken
#    - CUT FG by boundary -> CC
#    - For big merged blobs: LOCAL watershed inside blob
#    - Clean morph boundary overlay (NO Sobel)
# ==========================================================

import os
from pathlib import Path
import random, time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp


# -----------------------------
# 0) Config (EDIT HERE)
# -----------------------------
DATA_ROOT = "/root/workspace/DJF/PSCs/Unet++/data/perovskite"  # train/val/test each has images/ masks
TARGET_SIZE = (512, 512)     # (H,W)
CROP_BOTTOM = 0              # 建议裁掉底部标尺，比如 80~140
BATCH_SIZE = 2
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
SEED = 42

MASK_MODE = "red_on_black"   # "red_on_black" / "otsu"
PRETRAINED = True
THRESH_FOR_SAVING = 0.5

SAVE_DIR = "checkpoints_se_tversky_hybridcut"
BEST_NAME = "best.pth"

SPLIT_FOR_ABLATION = "val"   # "val" or "test"
VIS_NUM = 5
VIS_SEED = 0

# --- Semantic hysteresis thresholds (FG prob) ---
HYS_LOW = 0.35
HYS_HIGH = 0.60

# --- Instance split: Hybrid Morph params (最重要调参区) ---
# A) boundary-cut（粘连拆不开就：bnd_th ↓, bnd_close ↑, bnd_dilate ↑）
BND_TH = 0.15
BND_CLOSE = 5
BND_DILATE = 1

# B) local watershed for big blobs（切开后仍是大块才触发）
BIG_AREA = 1200
SEED_ERODE = 1
SEED_MIN_DIST = 5
SEED_PEAK_TH = 0.32
WS_W_DIST = 0.92
WS_W_BND  = 0.08

MIN_INSTANCE_AREA = 60
EDGE_THICKNESS = 1

# -----------------------------
# Manual fix for pred prob (可手动弥补概率图)
# -----------------------------
MANUAL_FIX_ENABLE = True

# 三种模式：
#   "file" : 读 manual_prob_fix/ 里的同名mask（推荐批处理/ablation用）
#   "gui"  : OpenCV窗口涂抹（需显示环境）
#   "mpl"  : Matplotlib交互涂抹（推荐Notebook/本地）
MANUAL_FIX_MODE = "mpl"              # "file" / "gui" / "mpl"
MANUAL_FIX_DIR  = "manual_prob_fix"  # 修正mask保存/读取目录

# 红色=强制更像FG；蓝色=强制更像BG
MANUAL_FG_FLOOR = 0.85
MANUAL_BG_CEIL  = 0.15

# 画笔
MANUAL_BRUSH = 10
MANUAL_AUTOSAVE = True              # mpl/gui 退出时自动保存mask到 MANUAL_FIX_DIR

os.environ.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)


# -----------------------------
# 1) Reproducibility
# -----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)


# -----------------------------
# 2) Utils: collect pairs
# -----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def collect_pairs(split_root: Path):
    img_dir = split_root / "images"
    msk_dir = split_root / "masks"
    if not img_dir.exists() or not msk_dir.exists():
        raise FileNotFoundError(f"Missing folders: {img_dir} or {msk_dir}")

    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if len(imgs) == 0:
        raise RuntimeError(f"No images in {img_dir}")

    pairs = []
    for ip in imgs:
        mp = msk_dir / ip.name
        if not mp.exists():
            cand = list(msk_dir.glob(ip.stem + ".*"))
            if len(cand) == 0:
                raise FileNotFoundError(f"Mask not found: {mp} (and no stem match)")
            mp = cand[0]
        pairs.append((ip, mp))
    return pairs


# -----------------------------
# 3) Mask reading (RED=FG)
# -----------------------------
def mask_red_on_black(mask_bgr: np.ndarray):
    hsv = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    fg = ((m1 | m2) > 0).astype(np.uint8)
    return fg

def load_mask_binary(mask_path: str, mode="red_on_black"):
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    if m.ndim == 3 and mode == "red_on_black":
        fg = mask_red_on_black(m)
        if fg.sum() > 0:
            return fg
        mode = "otsu"

    if m.ndim == 3:
        gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    else:
        gray = m
    gray = gray.astype(np.uint8)
    _, bin01 = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin01.astype(np.uint8)


# -----------------------------
# 3.5) Manual prob fix (红=补FG, 蓝=抹BG)
# -----------------------------
def _manual_fix_mask_from_bgr(mask_bgr: np.ndarray):
    """
    约定：
      红色 = 强制FG（prob拉高）
      蓝色 = 强制BG（prob压低）
    返回：add_fg(0/1), rm_bg(0/1)
    """
    hsv = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2HSV)

    # red
    r1 = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
    r2 = cv2.inRange(hsv, (170, 60, 60), (180, 255, 255))
    add_fg = ((r1 | r2) > 0).astype(np.uint8)

    # blue
    b1 = cv2.inRange(hsv, (95, 60, 60), (135, 255, 255))
    rm_bg = (b1 > 0).astype(np.uint8)

    return add_fg, rm_bg

def manual_fix_gui(prob: np.ndarray, name: str, brush: int = 10):
    """
    OpenCV画笔（有显示环境才好用）：
      左键画红=补FG
      右键画蓝=抹BG
      s保存并退出；q/ESC退出不保存
    返回：prob(原样)、canvas(BGR: red/blue)、saved(bool)
    """
    prob = prob.astype(np.float32)
    H, W = prob.shape

    canvas = np.zeros((H, W, 3), np.uint8)
    drawing = {"mode": None}  # "fg" or "bg"

    def _vis():
        base = (prob * 255).astype(np.uint8)
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        vis = base.copy()
        m = canvas.sum(axis=2) > 0
        vis[m] = (0.6 * vis[m] + 0.4 * canvas[m]).astype(np.uint8)
        cv2.putText(
            vis,
            f"{name}  (L=FG/red, R=BG/blue)  s=save  q=quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255,255,255), 2
        )
        return vis

    def on_mouse(event, x, y, flags, param):
        nonlocal canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["mode"] = "fg"
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing["mode"] = "bg"
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            drawing["mode"] = None

        if drawing["mode"] is not None and event in (
            cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN
        ):
            if drawing["mode"] == "fg":
                cv2.circle(canvas, (x, y), int(brush), (0, 0, 255), -1)  # red (BGR)
            else:
                cv2.circle(canvas, (x, y), int(brush), (255, 0, 0), -1)  # blue (BGR)

    win = "ManualProbFix(OpenCV)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    saved = False
    while True:
        cv2.imshow(win, _vis())
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')):  # ESC / q
            saved = False
            break
        if k == ord('s'):
            saved = True
            break

    cv2.destroyWindow(win)
    return prob.copy(), canvas, saved

def manual_fix_mpl(prob: np.ndarray, name: str, brush: int = 10):
    """
    Matplotlib交互涂抹（建议在 notebook 用 `%matplotlib qt` 或 `%matplotlib widget`）
    - 按键：
        f：FG(红)   b：BG(蓝)   e：橡皮擦
        ] / [ ：画笔变大 / 变小
        s：保存并退出   q/ESC：退出不保存
    - 鼠标：
        左键拖动涂抹
    返回：
      add_fg(0/1), rm_bg(0/1), canvas_bgr(BGR: red/blue), saved(bool)
    """
    prob = prob.astype(np.float32)
    H, W = prob.shape
    add_fg = np.zeros((H, W), np.uint8)
    rm_bg  = np.zeros((H, W), np.uint8)

    state = {
        "mode": "fg",   # fg/bg/erase
        "brush": int(brush),
        "down": False,
        "saved": False,
    }

    def _draw_circle(mask, x, y, r, v):
        cv2.circle(mask, (int(x), int(y)), int(r), int(v), -1)

    def _apply_brush(x, y):
        r = max(1, int(state["brush"]))
        if state["mode"] == "fg":
            _draw_circle(add_fg, x, y, r, 1)
            _draw_circle(rm_bg,  x, y, r, 0)
        elif state["mode"] == "bg":
            _draw_circle(rm_bg,  x, y, r, 1)
            _draw_circle(add_fg, x, y, r, 0)
        else:  # erase
            _draw_circle(add_fg, x, y, r, 0)
            _draw_circle(rm_bg,  x, y, r, 0)

    def _make_overlay_rgb():
        base = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        base_rgb = np.stack([base, base, base], axis=-1)

        ov = base_rgb.copy()
        mR = add_fg.astype(bool)
        ov[mR] = (0.55 * ov[mR] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)   # red (RGB)
        mB = rm_bg.astype(bool)
        ov[mB] = (0.55 * ov[mB] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)   # blue (RGB)
        return ov

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_title(
        f"{name}\n"
        f"mouse: L-drag paint | keys: f=FG(red) b=BG(blue) e=erase  [ / ]=brush  s=save  q=quit"
    )
    ax.axis("off")

    im = ax.imshow(_make_overlay_rgb())
    txt = ax.text(
        0.01, 0.01, "",
        transform=ax.transAxes,
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.35, pad=4)
    )

    def _refresh():
        im.set_data(_make_overlay_rgb())
        txt.set_text(f"mode={state['mode']}  brush={state['brush']}")
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            state["down"] = True
            if event.xdata is not None and event.ydata is not None:
                _apply_brush(event.xdata, event.ydata)
                _refresh()

    def on_release(event):
        if event.button == 1:
            state["down"] = False

    def on_move(event):
        if not state["down"]:
            return
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        _apply_brush(event.xdata, event.ydata)
        _refresh()

    def on_key(event):
        k = (event.key or "").lower()
        if k == "f":
            state["mode"] = "fg"
            _refresh()
        elif k == "b":
            state["mode"] = "bg"
            _refresh()
        elif k == "e":
            state["mode"] = "erase"
            _refresh()
        elif k == "]":
            state["brush"] = min(200, state["brush"] + 1)
            _refresh()
        elif k == "[":
            state["brush"] = max(1, state["brush"] - 1)
            _refresh()
        elif k == "s":
            state["saved"] = True
            plt.close(fig)
        elif k in ("q", "escape"):
            state["saved"] = False
            plt.close(fig)

    cid1 = fig.canvas.mpl_connect("button_press_event", on_press)
    cid2 = fig.canvas.mpl_connect("button_release_event", on_release)
    cid3 = fig.canvas.mpl_connect("motion_notify_event", on_move)
    cid4 = fig.canvas.mpl_connect("key_press_event", on_key)

    _refresh()
    plt.show()

    fig.canvas.mpl_disconnect(cid1)
    fig.canvas.mpl_disconnect(cid2)
    fig.canvas.mpl_disconnect(cid3)
    fig.canvas.mpl_disconnect(cid4)

    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[add_fg == 1] = (0, 0, 255)   # red in BGR
    canvas[rm_bg  == 1] = (255, 0, 0)   # blue in BGR
    return add_fg, rm_bg, canvas, state["saved"]

def apply_manual_prob_fix(prob: np.ndarray, name: str, target_hw, mode="file",
                          fix_dir="manual_prob_fix",
                          fg_floor=0.85, bg_ceil=0.15,
                          interactive: bool = False):
    """
    prob: (H,W) float32 0..1
    interactive:
      - True  : gui/mpl 模式会弹窗让你涂抹
      - False : 即使 mode=gui/mpl，也不会弹窗（只尝试读 fix_dir 的文件）
    """
    if not MANUAL_FIX_ENABLE:
        return prob

    H, W = target_hw
    prob = prob.astype(np.float32, copy=False)
    Path(fix_dir).mkdir(parents=True, exist_ok=True)

    def _apply_masks(add_fg, rm_bg):
        add_fg = cv2.resize(add_fg, (W, H), interpolation=cv2.INTER_NEAREST)
        rm_bg  = cv2.resize(rm_bg,  (W, H), interpolation=cv2.INTER_NEAREST)
        if add_fg.sum() > 0:
            prob[add_fg == 1] = np.maximum(prob[add_fg == 1], float(fg_floor))
        if rm_bg.sum() > 0:
            prob[rm_bg == 1] = np.minimum(prob[rm_bg == 1], float(bg_ceil))
        return np.clip(prob, 0.0, 1.0)

    # ---- 1) interactive mpl/gui ----
    if interactive and mode in ("mpl", "gui"):
        if mode == "mpl":
            add_fg, rm_bg, canvas_bgr, saved = manual_fix_mpl(prob, name, brush=MANUAL_BRUSH)
        else:
            _prob2, canvas_bgr, saved = manual_fix_gui(prob, name, brush=MANUAL_BRUSH)
            add_fg, rm_bg = _manual_fix_mask_from_bgr(canvas_bgr)

        prob_fixed = _apply_masks(add_fg, rm_bg)

        if MANUAL_AUTOSAVE and saved:
            outp = Path(fix_dir) / name
            cv2.imwrite(str(outp), canvas_bgr)
            print("[MANUAL] saved:", str(outp))

        return prob_fixed

    # ---- 2) file mode (or fallback when interactive=False) ----
    p = Path(fix_dir) / name
    if not p.exists():
        cand = list(Path(fix_dir).glob(Path(name).stem + ".*"))
        if len(cand) == 0:
            return prob
        p = cand[0]

    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        return prob

    if m.ndim == 2:
        add_fg = (m > 0).astype(np.uint8)
        rm_bg = np.zeros_like(add_fg, np.uint8)
    else:
        add_fg, rm_bg = _manual_fix_mask_from_bgr(m)

    return _apply_masks(add_fg, rm_bg)


# -----------------------------
# 4) Dataset
# -----------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

class SEMDataset(Dataset):
    """
    returns:
      x: (3,H,W) float32 ImageNet norm
      y: (1,H,W) float32 {0,1}  (FG=grains)
      raw: (H,W) uint8
      name: filename
    """
    def __init__(self, pairs, target_size=(512,512), mask_mode="red_on_black", crop_bottom=0):
        self.pairs = pairs
        self.target_size = target_size
        self.mask_mode = mask_mode
        self.crop_bottom = int(crop_bottom)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]

        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed reading image: {ip}")

        msk01 = load_mask_binary(str(mp), mode=self.mask_mode)  # 0/1

        if self.crop_bottom > 0:
            img = img[:-self.crop_bottom, :]
            msk01 = msk01[:-self.crop_bottom, :]

        th, tw = self.target_size
        img_rs = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        msk_rs = cv2.resize(msk01, (tw, th), interpolation=cv2.INTER_NEAREST)

        raw = img_rs.copy()
        img_f = img_rs.astype(np.float32) / 255.0

        x = np.stack([img_f, img_f, img_f], axis=0)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

        y = msk_rs.astype(np.float32)[None, ...]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(raw), ip.name

def sanity_show(ds, idx=0, title="SANITY"):
    x, y, raw, name = ds[idx]
    y_np = y[0].numpy().astype(np.uint8)
    print(f"[{title}] name={name} | y unique={np.unique(y_np)} | pos_ratio={y_np.mean():.4f}")

    rgb = np.stack([raw.numpy()]*3, axis=-1).astype(np.uint8)
    overlay = rgb.copy()
    overlay[y_np==1] = (0.5*overlay[y_np==1] + 0.5*np.array([255,0,0])).astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].imshow(raw.numpy(), cmap="gray"); axs[0].set_title("Raw"); axs[0].axis("off")
    axs[1].imshow(y_np, cmap="gray", vmin=0, vmax=1); axs[1].set_title("Mask(binary)"); axs[1].axis("off")
    axs[2].imshow(overlay); axs[2].set_title("Overlay (FG=red)"); axs[2].axis("off")
    plt.tight_layout(); plt.show()


# -----------------------------
# 5) Model: Unet++ + Encoder-SE
# -----------------------------
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w

class UnetPlusPlusEncoderSE(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        enc_w = "imagenet" if pretrained else None
        try:
            self.net = smp.UnetPlusPlus(
                encoder_name="resnet18",
                encoder_weights=enc_w,
                in_channels=3,
                classes=1,
                activation=None,
            )
        except Exception as e:
            print("[WARN] pretrained weights failed -> fallback None")
            print("       error:", repr(e))
            self.net = smp.UnetPlusPlus(
                encoder_name="resnet18",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None,
            )

        enc_ch = self.net.encoder.out_channels
        self.se = nn.ModuleList([
            nn.Identity() if i == 0 else SEBlock(c, reduction=16)
            for i, c in enumerate(enc_ch)
        ])

    def forward(self, x):
        feats = self.net.encoder(x)
        feats = [self.se[i](f) for i, f in enumerate(feats)]
        try:
            dec = self.net.decoder(*feats)
        except TypeError:
            dec = self.net.decoder(feats)
        return self.net.segmentation_head(dec)


# -----------------------------
# 6) Loss: Tversky (logits)
# -----------------------------
def tversky_loss_from_logits(logits, targets, alpha=0.3, beta=0.7, eps=1e-6):
    probs = torch.sigmoid(logits)
    tp = (probs * targets).sum(dim=(1,2,3))
    fp = (probs * (1 - targets)).sum(dim=(1,2,3))
    fn = ((1 - probs) * targets).sum(dim=(1,2,3))
    t = (tp + eps) / (tp + alpha*fp + beta*fn + eps)
    return 1.0 - t.mean()


# -----------------------------
# 7) Metrics
# -----------------------------
@torch.no_grad()
def dice_from_logits(logits, targets, th=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= th).float()
    inter = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets).sum(dim=(1,2,3))
    dice = (2*inter + eps) / (union + eps)
    return dice.mean().item()

def dice_iou_np(pred01, gt01, eps=1e-6):
    pred = pred01.astype(np.uint8)
    gt = gt01.astype(np.uint8)
    inter = float((pred & gt).sum())
    s1 = float(pred.sum()); s2 = float(gt.sum())
    dice = (2*inter + eps) / (s1 + s2 + eps)
    union = float((pred | gt).sum())
    iou = (inter + eps) / (union + eps)
    return dice, iou


# -----------------------------
# 8) Train / Save best / Load best
# -----------------------------
def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def train_one_epoch(model, loader, optimizer):
    model.train()
    total = 0.0
    for x, y, _raw, _name in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = tversky_loss_from_logits(logits, y, alpha=0.3, beta=0.7)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / max(1, len(loader.dataset))

@torch.no_grad()
def eval_val_dice(model, loader, th=0.5):
    model.eval()
    dices = []
    for x, y, _raw, _name in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        dices.append(dice_from_logits(logits, y, th=th))
    return float(np.mean(dices))

def train_and_get_best(train_loader, val_loader):
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    best_path = str((Path(SAVE_DIR) / BEST_NAME).resolve())

    seed_everything(SEED)
    model = UnetPlusPlusEncoderSE(pretrained=PRETRAINED).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_dice = -1.0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer)
        val_dice = eval_val_dice(model, val_loader, th=THRESH_FOR_SAVING)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_dice": best_dice}, best_path)

        print(f"[SE+Tversky] epoch {epoch:03d} | loss {tr_loss:.4f} | val Dice(th={THRESH_FOR_SAVING}) {val_dice:.4f} | best {best_dice:.4f}")

    print(f"Training done. time={(time.time()-t0):.1f}s")
    print("Best saved:", best_path, "best_dice=", best_dice)

    ckpt = safe_torch_load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model, best_path


# -----------------------------
# 9) Semantic binarization
# -----------------------------
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


# -----------------------------
# 12) Visualization
# -----------------------------
@torch.no_grad()
def visualize_samples(model, dataset, n=5, seed=0, title=""):
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    model.eval()
    for idx in idxs:
        x, y, raw, name = dataset[idx]
        x_b = x.unsqueeze(0).to(DEVICE)
        logits = model(x_b)
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()

        # ✅ 只在可视化阶段允许交互修正
        prob = apply_manual_prob_fix(
            prob, name=name, target_hw=prob.shape,
            mode=MANUAL_FIX_MODE,
            fix_dir=MANUAL_FIX_DIR,
            fg_floor=MANUAL_FG_FLOOR,
            bg_ceil=MANUAL_BG_CEIL,
            interactive=(MANUAL_FIX_MODE in ("mpl", "gui"))
        )

        gt = y[0].numpy().astype(np.uint8)
        img = raw.numpy().astype(np.uint8)

        pred_single = binarize_single(prob, 0.50)
        pred_hys    = binarize_hysteresis(prob, low=HYS_LOW, high=HYS_HIGH)

        lab, K, dbg = split_instances_hybrid_morph(
            fg_prob=prob,
            fg_mask01=pred_hys,
            bnd_th=BND_TH, bnd_close=BND_CLOSE, bnd_dilate=BND_DILATE,
            big_area=BIG_AREA,
            seed_erode=SEED_ERODE,
            seed_min_dist=SEED_MIN_DIST,
            seed_peak_th=SEED_PEAK_TH,
            w_dist=WS_W_DIST,
            w_bnd=WS_W_BND,
            min_instance_area=MIN_INSTANCE_AREA
        )

        inst_rgb = colorize_labels(lab, K+1)
        inst_edge_overlay = overlay_instance_boundaries_morph(img, lab, thickness=EDGE_THICKNESS)
        inst_overlay = overlay_instances_alpha(img, lab, alpha=0.45, seed=123)

        fig, axs = plt.subplots(2, 7, figsize=(30, 8))
        axs = axs.ravel()

        axs[0].imshow(img, cmap="gray"); axs[0].set_title(f"{title}\nRaw: {name}"); axs[0].axis("off")
        axs[1].imshow(gt, cmap="gray", vmin=0, vmax=1); axs[1].set_title("GT"); axs[1].axis("off")
        axs[2].imshow(prob, cmap="gray", vmin=0, vmax=1); axs[2].set_title("Pred prob (manual-fixed)"); axs[2].axis("off")
        axs[3].imshow(pred_single, cmap="gray", vmin=0, vmax=1); axs[3].set_title("Single th=0.50"); axs[3].axis("off")
        axs[4].imshow(pred_hys, cmap="gray", vmin=0, vmax=1); axs[4].set_title(f"Hysteresis {HYS_LOW}/{HYS_HIGH}"); axs[4].axis("off")
        axs[5].imshow(overlay_gray(img, pred_hys, (255,0,0))); axs[5].set_title("FG overlay"); axs[5].axis("off")
        axs[6].imshow(overlay_gray(img, gt, (0,255,0))); axs[6].set_title("GT overlay"); axs[6].axis("off")

        axs[7].imshow(dbg["bprob"], cmap="gray", vmin=0, vmax=1); axs[7].set_title("|∇p| boundary prob"); axs[7].axis("off")
        axs[8].imshow(dbg["bmask"], cmap="gray", vmin=0, vmax=1); axs[8].set_title("Boundary mask (close+dilate)"); axs[8].axis("off")
        axs[9].imshow(dbg["cut"], cmap="gray", vmin=0, vmax=1); axs[9].set_title("FG after CUT"); axs[9].axis("off")
        axs[10].imshow(inst_edge_overlay); axs[10].set_title(f"Instance boundaries (#={K})"); axs[10].axis("off")
        axs[11].imshow(inst_rgb); axs[11].set_title("Instances (color)"); axs[11].axis("off")
        axs[12].imshow(inst_overlay); axs[12].set_title("Instances overlay"); axs[12].axis("off")
        axs[13].axis("off")

        plt.tight_layout()
        plt.show()


# -----------------------------
# 13) Threshold ablation (semantic metrics)
# -----------------------------
@torch.no_grad()
def run_threshold_ablation(model, dataset, split_name="val"):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    variants = [
        ("single_th=0.50", lambda p: binarize_single(p, 0.50)),
        (f"hys_{HYS_LOW:.2f}_{HYS_HIGH:.2f}", lambda p: binarize_hysteresis(p, HYS_LOW, HYS_HIGH)),
        ("single_th=0.40", lambda p: binarize_single(p, 0.40)),
        ("single_th=0.60", lambda p: binarize_single(p, 0.60)),
    ]

    results = {name: {"dice": [], "iou": []} for name, _ in variants}

    model.eval()
    for x, y, _raw, _name in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        prob_batch = torch.sigmoid(logits)[:,0].cpu().numpy()
        gt_batch = y[:,0].cpu().numpy().astype(np.uint8)

        for b in range(prob_batch.shape[0]):
            p = prob_batch[b]

            # ✅ ablation 不弹窗，只读文件（即使 MANUAL_FIX_MODE=mpl/gui）
            p = apply_manual_prob_fix(
                p, name=_name[b], target_hw=p.shape,
                mode=MANUAL_FIX_MODE,
                fix_dir=MANUAL_FIX_DIR,
                fg_floor=MANUAL_FG_FLOOR,
                bg_ceil=MANUAL_BG_CEIL,
                interactive=False
            )

            g = gt_batch[b]
            for name, fn in variants:
                pred = fn(p)
                d, i = dice_iou_np(pred, g)
                results[name]["dice"].append(d)
                results[name]["iou"].append(i)

    def mean(a): return float(np.mean(a)) if len(a) else 0.0
    summary = [(name, mean(results[name]["dice"]), mean(results[name]["iou"])) for name in results]
    summary = sorted(summary, key=lambda x: x[1], reverse=True)

    print(f"\n===== Threshold Ablation on {split_name} (FG semantic) =====")
    print(f"{'variant':22s} | {'Dice':>7s} | {'IoU':>7s}")
    for name, d, i in summary:
        print(f"{name:22s} | {d:7.4f} | {i:7.4f}")

    best = summary[0][0]
    return variants, best


# -----------------------------
# 14) MAIN
# -----------------------------
def main():
    root = Path(DATA_ROOT)
    train_pairs = collect_pairs(root / "train")
    val_pairs   = collect_pairs(root / "val")
    test_root   = root / "test"
    test_pairs  = collect_pairs(test_root) if (test_root / "images").exists() else None

    print(f"[DATA] train={len(train_pairs)} | val={len(val_pairs)} | test={len(test_pairs) if test_pairs else 0}")
    print(f"[MASK] mode={MASK_MODE} | crop_bottom={CROP_BOTTOM}px")
    print("[INSTANCE PARAMS] "
          f"BND_TH={BND_TH}, BND_CLOSE={BND_CLOSE}, BND_DILATE={BND_DILATE}, "
          f"BIG_AREA={BIG_AREA}, SEED_MIN_DIST={SEED_MIN_DIST}, SEED_PEAK_TH={SEED_PEAK_TH}")
    print("[MANUAL FIX] "
          f"ENABLE={MANUAL_FIX_ENABLE}, MODE={MANUAL_FIX_MODE}, DIR={MANUAL_FIX_DIR}, "
          f"FG_FLOOR={MANUAL_FG_FLOOR}, BG_CEIL={MANUAL_BG_CEIL}, BRUSH={MANUAL_BRUSH}, AUTOSAVE={MANUAL_AUTOSAVE}")

    train_ds = SEMDataset(train_pairs, target_size=TARGET_SIZE, mask_mode=MASK_MODE, crop_bottom=CROP_BOTTOM)
    val_ds   = SEMDataset(val_pairs,   target_size=TARGET_SIZE, mask_mode=MASK_MODE, crop_bottom=CROP_BOTTOM)

    sanity_show(train_ds, 0, "TRAIN-SANITY")
    sanity_show(val_ds,   0, "VAL-SANITY")

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    # (1) Train + load best
    model, best_path = train_and_get_best(train_loader, val_loader)

    # (2) Choose split for ablation/visual
    if SPLIT_FOR_ABLATION == "val":
        ab_ds = val_ds
    elif SPLIT_FOR_ABLATION == "test":
        if test_pairs is None:
            raise RuntimeError("No test split found.")
        ab_ds = SEMDataset(test_pairs, target_size=TARGET_SIZE, mask_mode=MASK_MODE, crop_bottom=CROP_BOTTOM)
    else:
        raise ValueError("SPLIT_FOR_ABLATION must be 'val' or 'test'")

    # (3) Ablation (print)
    variants, best_name = run_threshold_ablation(model, ab_ds, split_name=SPLIT_FOR_ABLATION)
    print("[Ablation] best variant (by Dice):", best_name)

    # (4) Visualization (+ manual interactive fix)
    print("\n>>> Visualizing predictions + HYBRID instance split (boundary-cut + local WS)")
    visualize_samples(model, ab_ds, n=VIS_NUM, seed=VIS_SEED,
                      title="SE+Tversky + Hys + HybridCut(LocalWS)")

    print("\n[FINAL] best checkpoint saved at:", best_path)

if __name__ == "__main__":
    main()
