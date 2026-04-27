"""
PyTorch dataset for glaucoma triage — mixed binary + severity supervision.

Manifest CSV columns (v2):
    image_rid, image_path, binary_label, severity_label, label, split

    label — combined ground-truth used for severity training/eval:
            severity_label (1–4) if present, else 0 if binary_label==0, else NaN.

split values: "binary_train" | "binary_val" | "severity_train" | "severity_val" | "severity_test"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ── Keras VGG19 preprocessing ─────────────────────────────────────────────────

class KerasVGG19Preprocess:
    """
    Replicate tf.keras.applications.vgg19.preprocess_input exactly:
      1. PIL RGB → float32 [0, 255]
      2. Flip to BGR channel order
      3. Subtract BGR channel means [103.939, 116.779, 123.68]

    Replaces ToTensor + Normalize so all three methods (M1/M2/M3) use the same
    preprocessing as the original Keras model.
    """
    _MEAN = [103.939, 116.779, 123.68]  # BGR channel means

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # PIL RGB → float32 tensor (C, H, W), range [0, 255]
        x = TF.to_tensor(img) * 255.0
        # RGB → BGR
        x = x.flip(0)
        # Subtract BGR channel means
        mean = torch.tensor(self._MEAN, dtype=torch.float32).view(3, 1, 1)
        return x - mean


# ── Transforms ────────────────────────────────────────────────────────────────
# Exact match to the Keras model's preprocessing and augmentation strategy.
# best_hyperparameters.json: rotation_range=2, width_shift=0.041, height_shift=0.092,
#   zoom_range=0.033, horizontal_flip=True, vertical_flip=True, brightness_range≈0.007

TEST_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    KerasVGG19Preprocess(),
])

TRAIN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.RandomAffine(
        degrees=2,
        translate=(0.041, 0.092),
        scale=(0.967, 1.033),
    ),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.007),
    KerasVGG19Preprocess(),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class GlaucomaDataset(Dataset):
    """
    Mixed-supervision dataset.

    Every row returns (img, binary_label, severity_label, has_severity).
    Rows without a severity label return severity_label=-1, has_severity=False.
    Rows without a binary label  return binary_label=-1
      (these rows are skipped by the binary CE loss mask in train_M1/M3-Stage1).

    severity_fraction < 1.0 randomly masks severity labels to simulate the
    label-scarce regime studied in Exp 2 (scarcity sweep).
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "train",
        severity_fraction: float = 1.0,
        seed: int = 42,
    ) -> None:
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        # Severity scarcity subsampling (Exp 2 only — skip at fraction=1.0).
        # Stratified by severity class so rare classes (e.g. 3, 4) are not wiped
        # out at low fractions like 0.05 or 0.10.
        if severity_fraction < 1.0:
            rng = np.random.RandomState(seed)
            sev_rows = self.df[self.df["label"].notna()]
            keep_idx: list[int] = []
            for cls, group in sev_rows.groupby("label"):
                n_keep = max(1, int(len(group) * severity_fraction))
                chosen = rng.choice(group.index.tolist(), size=n_keep, replace=False)
                keep_idx.extend(chosen.tolist())
            keep_set = set(keep_idx)
            all_sev_idx = sev_rows.index.tolist()
            drop = [i for i in all_sev_idx if i not in keep_set]
            self.df.loc[drop, "label"] = np.nan

        self.transform = TRAIN_TRANSFORM if "train" in split else TEST_TRANSFORM

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)

        bl = row["binary_label"]
        sl = row["label"]
        binary_label   = int(bl) if pd.notna(bl) else -1
        has_severity   = bool(pd.notna(sl))
        severity_label = int(sl) if has_severity else -1

        return (
            img,
            torch.tensor(binary_label,   dtype=torch.long),
            torch.tensor(severity_label, dtype=torch.long),
            torch.tensor(has_severity,   dtype=torch.bool),
        )


# ── Convenience loader ────────────────────────────────────────────────────────

def load_splits(
    csv_path: str | Path,
    severity_fraction: float = 1.0,
    seed: int = 42,
) -> tuple[GlaucomaDataset, GlaucomaDataset, GlaucomaDataset]:
    """
    Return (train_ds, val_ds, test_ds).
    Val and test always use the full severity labels (fraction=1.0).
    """
    return (
        GlaucomaDataset(csv_path, "train", severity_fraction, seed),
        GlaucomaDataset(csv_path, "val",   1.0,               seed),
        GlaucomaDataset(csv_path, "test",  1.0,               seed),
    )
