import os
import csv
import math
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from Networks.network import ObjectDetectionModel
from loss_functions.loss_function import CombinedLoss
import matplotlib.pyplot as plt



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True



KITTI_CLASS_MAP = {
    'Car': 0, 'Van': 0, 'Truck': 0,
    'Pedestrian': 1, 'Person_sitting': 1,
    'Cyclist': 2,
}
NUM_CLASSES = 3
CLASS_NAMES = ['vehicle', 'pedestrian', 'cyclist']
BOX_DIM_NAMES = ['h', 'w', 'l', 'x', 'y', 'z', 'ry']


_BBOX_NORM_NP = np.array([5.0, 3.0, 10.0, 40.0, 3.0, 70.0, np.pi], dtype=np.float32)
_BBOX_NORM_T  = torch.from_numpy(_BBOX_NORM_NP)   # used on device in denormalize_boxes


def parse_first_valid_object(label_path):
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 15:
                continue
            cls_name = parts[0]
            if cls_name not in KITTI_CLASS_MAP:
                continue
            try:
                _, _, _, _, _, _, _, h, w, l, x, y, z, ry = map(float, parts[1:15])
            except ValueError:
                continue
            if h <= 0 or w <= 0 or l <= 0:
                continue
            return KITTI_CLASS_MAP[cls_name], (h, w, l, x, y, z, ry)
    return None


class KITTIDataset(Dataset):
    def __init__(self, velodyne_dir, label_dir, calib_dir, num_points=5000):
        self.velodyne_dir = velodyne_dir
        self.label_dir = label_dir
        self.calib_dir = calib_dir
        self.num_points = num_points

        self.files = []
        for f in sorted(os.listdir(velodyne_dir)):
            if not f.endswith('.bin'):
                continue
            file_id = f.split('.')[0]
            label_path = os.path.join(label_dir, file_id + '.txt')
            if not os.path.exists(label_path):
                continue
            if parse_first_valid_object(label_path) is not None:
                self.files.append(file_id)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_id = self.files[idx]
        calib = self.load_calibration(os.path.join(self.calib_dir, file_id + '.txt'))
        # Build combined velodyne→rectified-camera rotation+translation.
        Tr = calib['Tr_velo_to_cam'].reshape(3, 4)
        R0_flat = calib.get('R0_rect', np.eye(3, dtype=np.float32).ravel())
        R0 = R0_flat.reshape(3, 3)
        R = R0 @ Tr[:, :3]   # 3×3
        t = R0 @ Tr[:, 3]    # 3
        point_cloud = self.load_point_cloud(
            os.path.join(self.velodyne_dir, file_id + '.bin'), R, t)
        class_label, bbox = self.load_labels(
            os.path.join(self.label_dir, file_id + '.txt'))
        return point_cloud, class_label, bbox

    def load_point_cloud(self, file_path, R, t):
        pc = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        # Transform velodyne points into rectified camera frame.
        pc_cam = pc @ R.T + t                    # N×3
        pc_cam = pc_cam[pc_cam[:, 2] > 0]        # keep points in front of camera
        n = pc_cam.shape[0]
        if n == 0:
            pc_cam = np.zeros((self.num_points, 3), dtype=np.float32)
        elif n >= self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
            pc_cam = pc_cam[idx]
        else:
            idx = np.random.choice(n, self.num_points, replace=True)
            pc_cam = pc_cam[idx]
        return torch.from_numpy(pc_cam).float()

    def load_labels(self, file_path):
        cls_id, (h, w, l, x, y, z, ry) = parse_first_valid_object(file_path)
        raw = np.array([h, w, l, x, y, z, ry], dtype=np.float32)
        box = raw / _BBOX_NORM_NP
        return (torch.tensor(cls_id, dtype=torch.long),
                torch.from_numpy(box))

    def load_calibration(self, file_path):
        calib = {}
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    key, *values = line.split()
                    calib[key.rstrip(':')] = np.array(values, dtype=np.float32)
        return calib


def custom_collate_fn(batch):
    # All point clouds have exactly num_points points — no padding needed.
    point_clouds, class_labels, bbox_labels = zip(*batch)
    return (torch.stack(point_clouds),
            torch.stack(class_labels),
            torch.stack(bbox_labels))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def denormalize_boxes(boxes):
    norm = _BBOX_NORM_T.to(boxes.device)
    return boxes * norm


def pred_box8_to_7(pred_boxes_8):
    # Model outputs (h, w, l, x, y, z, sin_ry, cos_ry); collapse to (..., ry/π).
    sin_ry = pred_boxes_8[:, 6]
    cos_ry = pred_boxes_8[:, 7]
    ry_normalized = torch.atan2(sin_ry, cos_ry) / math.pi
    return torch.cat([pred_boxes_8[:, :6], ry_normalized.unsqueeze(1)], dim=1)


class MetricAccumulator:
    def __init__(self, num_classes=NUM_CLASSES, num_box_dims=7):
        self.num_classes = num_classes
        self.num_box_dims = num_box_dims
        self.reset()

    def reset(self):
        self.n = 0
        self.total_loss = 0.0
        self.cls_loss = 0.0
        self.reg_loss = 0.0
        self.iou_loss = 0.0
        self.iou_sum = 0.0
        self.iou_hits_05 = 0
        self.iou_hits_07 = 0
        self.correct = 0
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.box_abs_err_sum = np.zeros(self.num_box_dims, dtype=np.float64)
        self.grad_norms = []
        self.nan_batches = 0

    def update(self, batch_size, total_loss, info, pred_classes, true_classes,
               pred_boxes, true_boxes):
        self.n += batch_size
        self.total_loss += total_loss * batch_size
        self.cls_loss += info["cls_loss"].item() * batch_size
        self.reg_loss += info["reg_loss"].item() * batch_size
        self.iou_loss += info["iou_loss"].item() * batch_size

        iou = info["iou_per_sample"]
        self.iou_sum += iou.sum().item()
        self.iou_hits_05 += (iou > 0.5).sum().item()
        self.iou_hits_07 += (iou > 0.7).sum().item()

        preds = pred_classes.argmax(dim=1)
        self.correct += (preds == true_classes).sum().item()
        for t, p in zip(true_classes.tolist(), preds.tolist()):
            self.confusion[t, p] += 1

        # Per-axis MAE in real units
        abs_err = (denormalize_boxes(pred_boxes) - denormalize_boxes(true_boxes)).abs()
        self.box_abs_err_sum += abs_err.sum(dim=0).detach().cpu().numpy()

    def add_grad_norm(self, gn):
        self.grad_norms.append(gn)

    def add_nan(self):
        self.nan_batches += 1

    def summary(self):
        n = max(self.n, 1)
        diag = np.diag(self.confusion)
        col_sum = self.confusion.sum(axis=0)
        row_sum = self.confusion.sum(axis=1)
        precision = np.where(col_sum > 0, diag / np.maximum(col_sum, 1), 0.0)
        recall = np.where(row_sum > 0, diag / np.maximum(row_sum, 1), 0.0)
        return {
            "loss": self.total_loss / n,
            "cls_loss": self.cls_loss / n,
            "reg_loss": self.reg_loss / n,
            "iou_loss": self.iou_loss / n,
            "mean_iou": self.iou_sum / n,
            "iou_hit_0.5": self.iou_hits_05 / n,
            "iou_hit_0.7": self.iou_hits_07 / n,
            "accuracy": self.correct / n,
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "box_mae_meters": (self.box_abs_err_sum / n).tolist(),
            "grad_norm_mean": float(np.mean(self.grad_norms)) if self.grad_norms else 0.0,
            "nan_batches": self.nan_batches,
            "samples": self.n,
            "confusion": self.confusion.tolist(),
        }


def fmt_metrics(tag, s):
    box_mae = ", ".join(f"{n}={v:.3f}" for n, v in zip(BOX_DIM_NAMES, s["box_mae_meters"]))
    pr = " | ".join(
        f"{name} P={p:.2f}/R={r:.2f}"
        for name, p, r in zip(CLASS_NAMES, s["precision_per_class"], s["recall_per_class"])
    )
    return (f"[{tag}] loss={s['loss']:.4f} (cls={s['cls_loss']:.4f} "
            f"reg={s['reg_loss']:.4f} iou_l={s['iou_loss']:.4f}) | "
            f"acc={s['accuracy']:.3f} | mIoU={s['mean_iou']:.3f} "
            f"IoU@0.5={s['iou_hit_0.5']:.3f} IoU@0.7={s['iou_hit_0.7']:.3f}\n"
            f"       per-class: {pr}\n"
            f"       box MAE (m): {box_mae}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
velodyne_dir = 'kitti_3d_object_detection_dataset/training/velodyne'
label_dir = 'kitti_3d_object_detection_dataset/training/label_2'
calib_dir = 'kitti_3d_object_detection_dataset/training/calib'
output_dir = 'outputs_run6_200epochs'
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

full_dataset = KITTIDataset(velodyne_dir, label_dir, calib_dir)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)
print(f"Dataset: {len(full_dataset)} | train={train_size} val={val_size}")

BATCH_SIZE = 64
NUM_WORKERS = 16
PIN_MEMORY = (device.type == 'cuda')

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
)

model = ObjectDetectionModel(num_classes=NUM_CLASSES, feature_dim=64).to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters (head only): {trainable:,}")

criterion = CombinedLoss(alpha=1.0, beta=1.0).to(device)

# Backbone (PointNet++) starts frozen; only head params are in the optimizer.
# After warmup it is unfrozen into a separate param group at BACKBONE_LR.
# LR is sqrt-scaled from 1e-3 @ bs=32 → 1.4e-3 @ bs=64.
HEAD_LR       = 1.4e-3
BACKBONE_LR   = 1e-4
num_epochs    = 200
warmup_epochs = max(1, num_epochs // 10)   # 5 epochs

_head_params = [p for n, p in model.named_parameters() if not n.startswith('pointnet')]
optimizer = optim.Adam(_head_params, lr=HEAD_LR)

scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                    total_iters=warmup_epochs),
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs)),
    ],
    milestones=[warmup_epochs],
)
GRAD_CLIP = 1.0

# Mixed precision (Blackwell tensor cores). Disabled automatically on CPU.
USE_AMP = False
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
print(f"Batch size: {BATCH_SIZE} | workers: {NUM_WORKERS} | "
      f"AMP: {USE_AMP} | pin_memory: {PIN_MEMORY}")

best_val_iou = -1.0
best_model_path = 'best_object_detection_model_run6_200epochs.pth'

# CSV header
metrics_csv_path = os.path.join(output_dir, 'metrics.csv')
csv_header = ['epoch', 'lr', 'epoch_seconds', 'samples_per_sec',
              'grad_norm_mean', 'nan_batches', 'peak_gpu_mb']
for split in ('train', 'val'):
    csv_header += [f'{split}_loss', f'{split}_cls_loss', f'{split}_reg_loss',
                   f'{split}_iou_loss', f'{split}_acc', f'{split}_mean_iou',
                   f'{split}_iou_hit_0.5', f'{split}_iou_hit_0.7']
    for cn in CLASS_NAMES:
        csv_header += [f'{split}_precision_{cn}', f'{split}_recall_{cn}']
    for bn in BOX_DIM_NAMES:
        csv_header += [f'{split}_box_mae_{bn}']

with open(metrics_csv_path, 'w', newline='') as f:
    csv.writer(f).writerow(csv_header)

# Per-epoch history for plotting
history = {"epoch": [], "train": [], "val": [], "lr": []}

for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

    # Unfreeze backbone at the end of warmup so the head has a head-start.
    # Only sa1/sa2 are used in forward(), so only those need optimizer state.
    if epoch == warmup_epochs:
        backbone_modules = (model.pointnet.sa1, model.pointnet.sa2)
        for module in backbone_modules:
            for param in module.parameters():
                param.requires_grad_(True)
        backbone_params = []
        for module in backbone_modules:
            backbone_params.extend(module.parameters())
        optimizer.add_param_group({"params": backbone_params, "lr": BACKBONE_LR})
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [backbone sa1+sa2 unfrozen] lr={BACKBONE_LR} | total trainable={total:,}")

    # Bump IoU loss weight once IoU is non-trivially > 0 so it dominates SmoothL1.
    if epoch == 10:
        criterion.beta = 3.0
        print(f"  [iou loss weight bumped to {criterion.beta}]")

    epoch_start = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # ---- Train ----
    model.train()
    train_metrics = MetricAccumulator()
    for i, (inputs, class_labels, bbox_labels) in enumerate(train_loader):
        # Sanity-check inputs BEFORE moving to device, so a bad sample fails on CPU
        # with a clean Python traceback instead of an asynchronous CUDA assert.
        assert class_labels.dtype == torch.long, f"class_labels dtype={class_labels.dtype}"
        assert class_labels.min().item() >= 0 and class_labels.max().item() < NUM_CLASSES, \
            f"class label out of range [0,{NUM_CLASSES}): " \
            f"min={class_labels.min().item()} max={class_labels.max().item()}"
        assert torch.isfinite(bbox_labels).all().item(), "non-finite bbox label in batch"
        assert torch.isfinite(inputs).all().item(), "non-finite point cloud in batch"

        inputs = inputs.to(device, non_blocking=True)
        class_labels = class_labels.to(device, non_blocking=True)
        bbox_labels = bbox_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            class_outputs, bbox_outputs = model(inputs)
            loss, info = criterion(bbox_outputs, bbox_labels, class_outputs, class_labels)

        if not torch.isfinite(loss):
            train_metrics.add_nan()
            print(f"  [iter {i + 1}] non-finite loss, skipping batch")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        train_metrics.add_grad_norm(float(grad_norm))
        scaler.step(optimizer)
        scaler.update()

        bs = inputs.size(0)
        pred_boxes_7 = pred_box8_to_7(bbox_outputs.detach().float())
        train_metrics.update(bs, loss.item(), info,
                             class_outputs.detach().float(), class_labels,
                             pred_boxes_7, bbox_labels)

        if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
            print(f"  iter {i + 1}/{len(train_loader)} | "
                  f"loss={loss.item():.4f} | grad_norm={float(grad_norm):.3f}",
                  flush=True)

    # ---- Validate ----
    model.eval()
    val_metrics = MetricAccumulator()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_AMP):
        for inputs, class_labels, bbox_labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)
            bbox_labels = bbox_labels.to(device, non_blocking=True)

            class_outputs, bbox_outputs = model(inputs)
            v_loss, v_info = criterion(bbox_outputs, bbox_labels, class_outputs, class_labels)
            if not torch.isfinite(v_loss):
                val_metrics.add_nan()
                continue
            val_pred_boxes_7 = pred_box8_to_7(bbox_outputs.float())
            val_metrics.update(inputs.size(0), v_loss.item(), v_info,
                               class_outputs.float(), class_labels,
                               val_pred_boxes_7, bbox_labels)

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    epoch_seconds = time.perf_counter() - epoch_start
    samples_per_sec = (train_metrics.n + val_metrics.n) / max(epoch_seconds, 1e-9)
    peak_gpu_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)
                   if device.type == 'cuda' else 0.0)

    train_s = train_metrics.summary()
    val_s = val_metrics.summary()

    print(f"  epoch_time={epoch_seconds:.1f}s | samples/s={samples_per_sec:.1f} | "
          f"lr={current_lr:.2e} | grad_norm_mean={train_s['grad_norm_mean']:.3f} | "
          f"peak_gpu={peak_gpu_mb:.0f} MB | "
          f"nan_batches: train={train_s['nan_batches']} val={val_s['nan_batches']}")
    print(fmt_metrics("train", train_s))
    print(fmt_metrics("val  ", val_s))

    # CSV row
    row = [epoch + 1, current_lr, epoch_seconds, samples_per_sec,
           train_s['grad_norm_mean'],
           train_s['nan_batches'] + val_s['nan_batches']]
    for s in (train_s, val_s):
        row += [s['loss'], s['cls_loss'], s['reg_loss'], s['iou_loss'],
                s['accuracy'], s['mean_iou'], s['iou_hit_0.5'], s['iou_hit_0.7']]
        row += list(s['precision_per_class']) + list(s['recall_per_class'])
        # interleave precision/recall per class to match header
        # header was [precision_c0, recall_c0, precision_c1, recall_c1, ...] — fix below
    # Rebuild row to match header order exactly:
    row = [epoch + 1, current_lr, epoch_seconds, samples_per_sec,
           train_s['grad_norm_mean'],
           train_s['nan_batches'] + val_s['nan_batches'],
           peak_gpu_mb]
    for s in (train_s, val_s):
        row += [s['loss'], s['cls_loss'], s['reg_loss'], s['iou_loss'],
                s['accuracy'], s['mean_iou'], s['iou_hit_0.5'], s['iou_hit_0.7']]
        for ci in range(NUM_CLASSES):
            row += [s['precision_per_class'][ci], s['recall_per_class'][ci]]
        row += list(s['box_mae_meters'])
    with open(metrics_csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(row)

    history["epoch"].append(epoch + 1)
    history["train"].append(train_s)
    history["val"].append(val_s)
    history["lr"].append(current_lr)

    if val_s['mean_iou'] > best_val_iou:
        best_val_iou = val_s['mean_iou']
        torch.save(model.state_dict(), best_model_path)
        print(f"  ** new best mean IoU on val: {best_val_iou:.4f} -> saved {best_model_path}")

print(f"\nTraining complete. Best val mean IoU: {best_val_iou:.4f}")
print(f"Per-epoch metrics: {metrics_csv_path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _series(split, key):
    return [h[key] for h in history[split]]


epochs = history["epoch"]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

axes[0, 0].plot(epochs, _series("train", "loss"), label='train total')
axes[0, 0].plot(epochs, _series("val", "loss"), label='val total')
axes[0, 0].plot(epochs, _series("train", "cls_loss"), '--', label='train cls')
axes[0, 0].plot(epochs, _series("train", "reg_loss"), '--', label='train reg')
axes[0, 0].plot(epochs, _series("train", "iou_loss"), '--', label='train iou_l')
axes[0, 0].set_title('Loss components')
axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()

axes[0, 1].plot(epochs, _series("train", "mean_iou"), label='train mIoU')
axes[0, 1].plot(epochs, _series("val", "mean_iou"), label='val mIoU')
axes[0, 1].plot(epochs, _series("train", "iou_hit_0.5"), '--', label='train IoU@0.5')
axes[0, 1].plot(epochs, _series("val", "iou_hit_0.5"), '--', label='val IoU@0.5')
axes[0, 1].set_title('Detection quality')
axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('IoU / hit-rate')
axes[0, 1].legend()

axes[1, 0].plot(epochs, _series("train", "accuracy"), label='train acc')
axes[1, 0].plot(epochs, _series("val", "accuracy"), label='val acc')
axes[1, 0].set_title('Classification accuracy')
axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()

# Per-axis box MAE on validation
for i, name in enumerate(BOX_DIM_NAMES):
    axes[1, 1].plot(epochs, [h["box_mae_meters"][i] for h in history["val"]], label=name)
axes[1, 1].set_title('Validation box MAE (meters / radians)')
axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Abs error')
axes[1, 1].legend(ncol=2, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=150)
plt.show()
