"""
=============================================================================
Driver Drowsiness Detection Using MobileViT (PyTorch + timm)
=============================================================================
Dataset  : kagglehub  talhabhatti7262/drivers-drowsiness-detection
Model    : mobilevitv2_200 image classifier
Task     : Binary classification - 0 = Drowsy, 1 = Alert
=============================================================================

Install dependencies before running:
    pip install torch torchvision timm opencv-python-headless \
                scikit-learn matplotlib seaborn tqdm Pillow

Run:
    python drowsiness_detection.py
=============================================================================
"""

import os
import time
import random
import warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Hyperparameters ─────────────────────────────────────────────────────────────────
CONFIG = {
    # Data
    "img_size":        224,
    # Model
    "cnn_backbone":    "mobilenetv3_large_100",
    "num_classes":     2,
    "dropout":         0.4,
    # Training
    "batch_size":      64,
    "epochs":          3,
    "learning_rate":   1e-4,
    "weight_decay":    1e-5,
    "patience":        3,
    # Deployment
    "alert_threshold": 0.7,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    # Paths
    "output_dir":      "outputs",
    # Webcam performance
    "camera_preview_width":  1280,
    "camera_preview_height": 720,
    "camera_infer_interval": 2,
}


os.makedirs(CONFIG["output_dir"], exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print(f"[Config] Device : {CONFIG['device']}")
print(f"[Config] Hyperparameters:\n"
      f"         batch_size={CONFIG['batch_size']}, "
      f"epochs={CONFIG['epochs']}, "
      f"lr={CONFIG['learning_rate']}, "
      f"img_size={CONFIG['img_size']}, "
      f"dropout={CONFIG['dropout']}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])


def detect_and_crop_face(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Detect the largest face in a BGR frame and return a cropped RGB patch.
    Supports pre-cropped eye images by passing them through directly if small.
    Returns None if a large frame is passed but no face is detected.
    """
    if frame_bgr is None:
        return None
        
    h, w = frame_bgr.shape[:2]
    if h <= 150 or w <= 150:
        # Pre-cropped eye image (from dataset), pass through directly
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    pad = int(0.10 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_bgr.shape[1], x + w + pad)
    y2 = min(frame_bgr.shape[0], y + h + pad)
    crop = frame_bgr[y1:y2, x1:x2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)



def extract_frame_from_video(video_path: str) -> np.ndarray:
    """
    Extract a single representative frame from a video file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    mid_frame = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None

    return detect_and_crop_face(frame)


def preprocess_image(frame_rgb: np.ndarray) -> torch.Tensor:
    """
    Convert a raw RGB numpy array -> normalised tensor [3, H, W].
    """
    img = Image.fromarray(frame_rgb)
    return FRAME_TRANSFORM(img)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dataset Loader
# ─────────────────────────────────────────────────────────────────────────────

def build_file_label_list(dataset_root: str):
    """
    Walk `dataset_root` and collect (file_path, label) pairs.
    """
    VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

    items = []
    ALERT_KEYS  = {"alert", "awake", "non_drowsy", "nondrowsy", "active", "open", "openeye"}
    DROWSY_KEYS = {"drowsy", "sleep", "sleepy", "yawn", "fatigue", "tired", "close", "closeeye"}


    for root, dirs, files in os.walk(dataset_root):
        folder_name = Path(root).name.lower()
        if any(k in folder_name for k in ALERT_KEYS):
            label = 0
        elif any(k in folder_name for k in DROWSY_KEYS):
            label = 1
        else:
            continue

        for fname in files:
            ext = Path(fname).suffix.lower()
            fpath = os.path.join(root, fname)
            if ext in VIDEO_EXT:
                items.append((fpath, label, True))
            elif ext in IMAGE_EXT:
                items.append((fpath, label, False))

    random.shuffle(items)
    print(f"[Dataset] Found {len(items)} files  "
          f"(alert={sum(1 for _, l, _ in items if l==0)}, "
          f"drowsy={sum(1 for _, l, _ in items if l==1)})")
    return items


class DrowsinessImageDataset(Dataset):
    """
    Loads a single RGB image tensor from a video clip or image file.
    """

    def __init__(self, items: list):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, is_video = self.items[idx]
        if is_video:
            frame = extract_frame_from_video(path)
        else:
            img = cv2.imread(path)
            frame = detect_and_crop_face(img) if img is not None else None

        if frame is None:
            frame = np.zeros((CONFIG["img_size"], CONFIG["img_size"], 3), dtype=np.uint8)

        img_tensor = preprocess_image(frame)
        return img_tensor, torch.tensor(label, dtype=torch.long)


def make_dataloaders(dataset_root: str):
    if dataset_root and os.path.isdir(dataset_root):
        items = build_file_label_list(dataset_root)
    else:
        items = []

    if len(items) < 10:
        print("[Dataset] Using synthetic data for demonstration ...")
        items = _make_synthetic_items(n=200)
    elif len(items) > 8000:
        print("[Dataset] Dataset is large (%d files). Sub-sampling to 8,000 balanced files for fast and highly accurate training..." % len(items))
        random.seed(42)
        random.shuffle(items)
        items = items[:8000]


    train_items, test_items = train_test_split(
        items, test_size=0.15, random_state=42,
        stratify=[l for _, l, _ in items]
    )
    train_items, val_items = train_test_split(
        train_items, test_size=0.15, random_state=42,
        stratify=[l for _, l, _ in train_items]
    )

    def make_loader(split_items, shuffle):
        ds = DrowsinessImageDataset(split_items)
        return DataLoader(
            ds, batch_size=CONFIG["batch_size"],
            shuffle=shuffle, num_workers=0, pin_memory=False,
        )

    loaders = {
        "train": make_loader(train_items, shuffle=True),
        "val":   make_loader(val_items,   shuffle=False),
        "test":  make_loader(test_items,  shuffle=False),
    }
    print(f"[Dataset] Split -> train={len(train_items)}, "
          f"val={len(val_items)}, test={len(test_items)}")
    return loaders


def _make_synthetic_items(n: int = 200):
    tmp = Path("/tmp/synthetic_drowsy")
    for cls, lbl in [("alert", 0), ("drowsy", 1)]:
        (tmp / cls).mkdir(parents=True, exist_ok=True)

    items = []
    for i in range(n):
        label = i % 2
        cls = "alert" if label == 0 else "drowsy"
        fpath = str(tmp / cls / f"frame_{i:04d}.jpg")
        if not os.path.exists(fpath):
            colour = (100, 200, 100) if label == 0 else (200, 80, 80)
            img = np.full((64, 64, 3), colour, dtype=np.uint8)
            cv2.imwrite(fpath, img)
        items.append((fpath, label, False))

    return items

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model Architecture: MobileViT Image Classifier
# ─────────────────────────────────────────────────────────────────────────────

from ultralytics import YOLO

class DrowsinessDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained YOLOv11 classification model
        self.model = YOLO("best.pt")

    def to(self, device):
        # YOLO handles its own device placement during predict()
        self.model.to(device)
        return self

    def eval(self):
        return self

    def forward(self, x):
        return self.model(x)



def model_summary(model: nn.Module, device: str):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - train
    print(f"\n[Model] Architecture: {CONFIG['cnn_backbone']} image classifier")
    print(f"        Total params   : {total:,}")
    print(f"        Trainable      : {train:,}")
    print(f"        Frozen (none)  : {frozen:,}")
    print(f"        Device         : {device}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []

    for imgs, labels in tqdm(loader, desc="  Eval ", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_labels),
        np.array(all_probs),
    )


def train(model: nn.Module, loaders: dict, device: str) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
    )

    history = defaultdict(list)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"\n[Train] Starting - {CONFIG['epochs']} epochs, "
          f"patience={CONFIG['patience']}")
    print("-" * 60)

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        va_loss, va_acc, _, _ = evaluate(
            model, loaders["val"], criterion, device
        )
        scheduler.step()

        history["tr_loss"].append(tr_loss)
        history["tr_acc"].append(tr_acc)
        history["va_loss"].append(va_loss)
        history["va_acc"].append(va_acc)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{CONFIG['epochs']}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  "
              f"({elapsed:.1f}s)")

        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"  [EarlyStopping] No improvement for "
                      f"{CONFIG['patience']} epochs - stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print("[Train] Best weights restored.")

    ckpt_path = os.path.join(CONFIG["output_dir"], "best_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Train] Checkpoint saved -> {ckpt_path}")

    return dict(history)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Evaluation & Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float = CONFIG["alert_threshold"],
) -> dict:
    criterion = nn.CrossEntropyLoss()
    _, _, y_true, y_prob = evaluate(model, loader, criterion, device)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Threshold   : {threshold}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall      : {rec:.4f}")
    print(f"  F1-score    : {f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Drowsy", "Alert"],
                                zero_division=0))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred Drowsy", "Pred Alert"],
        yticklabels=["True Drowsy", "True Alert"],
        ax=ax, linewidths=0.5,
    )
    ax.set_title("Confusion Matrix", fontsize=13, pad=10)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    cm_path = os.path.join(CONFIG["output_dir"], "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n[Eval] Confusion matrix saved -> {cm_path}")

    return {
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1,
        "confusion_matrix": cm,
        "y_true": y_true, "y_prob": y_prob,
    }


def plot_training_curves(history: dict):
    epochs = range(1, len(history["tr_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["tr_loss"], label="Train Loss",  color="#3B8BD4")
    axes[0].plot(epochs, history["va_loss"], label="Val Loss",    color="#E24B4A", linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["tr_acc"], label="Train Acc",   color="#1D9E75")
    axes[1].plot(epochs, history["va_acc"], label="Val Acc",     color="#EF9F27", linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training Curves - MobileViT Drowsiness Detector", fontsize=12)
    plt.tight_layout()
    curve_path = os.path.join(CONFIG["output_dir"], "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"[Eval] Training curves saved -> {curve_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Inference Pipeline & Latency Analysis
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_single_image(
    model: nn.Module,
    frame: np.ndarray,
    device: str,
    threshold: float = CONFIG["alert_threshold"],
) -> dict:
    if frame is None or frame.size == 0:
        return {
            "probability": 0.0,
            "prediction": "No Face Detected",
            "alert": False,
            "latency_ms": 0.0,
        }
        
    start = time.perf_counter()
    
    # Run YOLO classification inference
    # Note: Ultralytics YOLO model.predict() automatically handles conversion and transfers to device
    results = model.model.predict(frame, device=device, verbose=False)
    
    if hasattr(device, 'type') and device.type == "cuda":
        torch.cuda.synchronize()
    latency = (time.perf_counter() - start) * 1000
    
    probs = results[0].probs
    
    # Class 0: 'Drowsy', Class 1: 'Non Drowsy'
    prob_drowsy = float(probs.data[0].item())
    
    alert = prob_drowsy >= threshold
    prediction = "Drowsy" if alert else "Alert"
    
    return {
        "probability": round(prob_drowsy, 4),
        "prediction": prediction,
        "alert": alert,
        "latency_ms": round(latency, 2),
    }


def benchmark_latency(
    model: nn.Module,
    device: str,
    n_runs: int = 50,
    warmup: int = 5,
) -> dict:
    model.eval()
    C, H, W = 3, CONFIG["img_size"], CONFIG["img_size"]
    latencies = []

    with torch.no_grad():
        dummy = torch.randn(1, C, H, W, device=device)

        for _ in range(warmup):
            _ = model(dummy)
            if hasattr(device, 'type') and device.type == "cuda":
                torch.cuda.synchronize()

        for _ in range(n_runs):
            if hasattr(device, 'type') and device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if hasattr(device, 'type') and device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    stats = {
        "mean_ms": round(float(arr.mean()), 2),
        "std_ms":  round(float(arr.std()),  2),
        "p50_ms":  round(float(np.percentile(arr, 50)), 2),
        "p95_ms":  round(float(np.percentile(arr, 95)), 2),
        "p99_ms":  round(float(np.percentile(arr, 99)), 2),
        "n_runs":  n_runs,
    }

    fps_estimate = 1000.0 / stats["mean_ms"] if stats["mean_ms"] > 0 else 0.0
    print("\n[Latency] Inference Benchmark")
    print(f"          Runs    : {n_runs}")
    print(f"          Mean    : {stats['mean_ms']} ms")
    print(f"          Std     : {stats['std_ms']} ms")
    print(f"          P50     : {stats['p50_ms']} ms")
    print(f"          P95     : {stats['p95_ms']} ms")
    print(f"          P99     : {stats['p99_ms']} ms")
    print(f"          Est FPS : {fps_estimate:.1f} "
          f"({'feasible' if fps_estimate >= 5 else 'too slow'} for real-time @ 5 fps)")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(arr, bins=20, color="#3B8BD4", edgecolor="white", alpha=0.85)
    ax.axvline(stats["mean_ms"], color="#E24B4A", linestyle="--",
               label=f"Mean = {stats['mean_ms']} ms")
    ax.axvline(stats["p95_ms"],  color="#EF9F27", linestyle=":",
               label=f"P95  = {stats['p95_ms']} ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Inference Latency Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    lat_path = os.path.join(CONFIG["output_dir"], "latency_analysis.png")
    plt.savefig(lat_path, dpi=150)
    plt.close()
    print(f"[Latency] Distribution saved -> {lat_path}")

    return stats


def run_webcam_inference(model: nn.Module, device: str, camera_id: int = 0):
    print("\n[Webcam] Starting real-time inference...")
    print("         Press 'q' to quit")
    print("-" * 60)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Webcam] ERROR: Could not open camera id {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_preview_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_preview_height"])

    frame_count = 0
    prediction_text = "Loading..."
    confidence = 0.0
    alert_triggered = False
    latency_ms = 0.0
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0

    model.eval()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("[Webcam] Failed to read frame")
                break

            frame_count += 1
            face_crop_rgb = detect_and_crop_face(frame_bgr)

            if frame_count % CONFIG["camera_infer_interval"] == 0:
                result = infer_single_image(model, face_crop_rgb, device)
                prediction_text = result["prediction"]
                confidence = result["probability"]
                alert_triggered = result["alert"]
                latency_ms = result["latency_ms"]

            display_frame = frame_bgr.copy()
            h, w = display_frame.shape[:2]
            box_height = 80
            cv2.rectangle(display_frame, (0, 0), (w, box_height),
                         (0, 0, 0), -1)

            color = (0, 0, 255) if alert_triggered else (0, 255, 0)
            text = f"Prediction: {prediction_text}"
            cv2.putText(display_frame, text, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

            info_text = f"Confidence: {confidence:.2%}  |  Latency: {latency_ms} ms"
            cv2.putText(display_frame, info_text, (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            fps_counter += 1
            if time.time() - fps_timer > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()

            fps_text = f"FPS: {current_fps}"
            cv2.putText(display_frame, fps_text, (w - 150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            if alert_triggered:
                cv2.rectangle(display_frame, (5, 5), (35, 35), (0, 0, 255), -1)
                cv2.circle(display_frame, (20, 20), 8, (0, 255, 255), 3)

            cv2.imshow("Drowsiness Detection", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[Webcam] Quit requested by user")
                break

    except KeyboardInterrupt:
        print("\n[Webcam] Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Webcam] Camera released and display closed")


def run_demo_inference(model: nn.Module, device: str):
    print("[Demo] Simulated real-time inference on 5 random samples")
    print("-" * 60)

    for i in range(5):
        frames = [
            np.random.randint(0, 255, (CONFIG["img_size"], CONFIG["img_size"], 3), dtype=np.uint8)
        ]
        result = infer_single_image(model, frames[0], device)

        alert_str = "!  ALERT TRIGGERED" if result["alert"] else "   OK"
        print(
            f"  Sample {i+1}:  prob={result['probability']:.4f}  "
            f"pred={result['prediction']:7s}  "
            f"latency={result['latency_ms']} ms  {alert_str}"
        )
    print("-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = CONFIG["device"]

    print("\n" + "=" * 60)
    print("  STEP 1 - Dataset")
    print("=" * 60)
    dataset_root = r"C:\Users\sivas\.cache\kagglehub\datasets\talhabhatti7262\drivers-drowsiness-detection\versions\1\Final\PreparedData"


    print("\n" + "=" * 60)
    print("  STEP 2 - Preprocessing & DataLoaders")
    print("=" * 60)
    loaders = make_dataloaders(dataset_root)

    print("\n" + "=" * 60)
    print("  STEP 3 - Model Architecture")
    print("=" * 60)
    model = DrowsinessDetector().to(device)
    model_summary(model, device)

    print("\n" + "=" * 60)
    print("  STEP 4 - Training")
    print("=" * 60)
    history = train(model, loaders, device)
    plot_training_curves(history)

    print("\n" + "=" * 60)
    print("  STEP 5 - Evaluation")
    print("=" * 60)
    results = full_evaluation(model, loaders["test"], device)

    print("\n" + "=" * 60)
    print("  STEP 6 - Latency Analysis")
    print("=" * 60)
    benchmark_latency(model, device, n_runs=50)

    # print("\n" + "=" * 60)
    # print("  STEP 7 - Real-time Webcam Inference")
    # print("=" * 60)
    # run_webcam_inference(model, device)



if __name__ == "__main__":
    main()
