"""
Deriva-ML data pipeline v2 for glaucoma triage project.

Downloads 5 study datasets (v2) and builds manifest.csv with three label columns:

    image_rid, image_path, binary_label, severity_label, label, split

Columns:
    binary_label   — 0/1 from Image_Diagnosis (Initial Diagnosis only); NaN if absent
    severity_label — 1–4 from clinical records chain; NaN if absent
    label          — combined ground-truth: severity_label (1–4) if present,
                     else 0 if binary_label==0, else NaN.
                     This is the target for M2/M3 severity training and evaluation.
    split          — 'binary_train' | 'binary_val' | 'severity_train' |
                     'severity_val' | 'severity_test'

Study datasets v2 (see docs/CATALOG_NOTES.md):
    5-Z9GT — binary_train   (4,212 images)
    5-ZHRE — binary_val     (1,404 images)
    5-ZQ8P — severity_train (2,236 images: 1,118 severity + 1,118 grade-0)
    5-ZVMT — severity_val   (558 images:   279 severity + 279 grade-0)
    5-ZWR2 — severity_test  (686 images:   343 severity + 343 grade-0)

Key differences from data_pipeline.py (v1):
    - Initial Diagnosis only — Expert_Consensus rows are ignored everywhere
    - Three label columns instead of two
    - No per-subject dedup (datasets are already curated)
    - Rows with neither binary_label nor severity_label are dropped

Bag structure (Image-based datasets):
    data/Dataset/Dataset_Image/Image.csv
    data/Dataset/Dataset_Image/Image/Observation/
        Clinical_Records_Observation/Clinical_Records/
        Execution_Clinical_Records_Glaucoma_Severity.csv  ← severity labels
    data/Dataset/Dataset_Image/Image/Image_Diagnosis.csv  ← binary labels
    data/asset/{RID}/Image/{Filename}                     ← image files

Usage
-----
    python src/catalog/data_pipeline_v2.py --output-dir data/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Study dataset RIDs (v2 — created by construct_datasets_v2.py)
# ---------------------------------------------------------------------------

STUDY_DATASETS = {
    "binary_train":   "5-Z9GT",
    "binary_val":     "5-ZHRE",
    "severity_train": "5-ZQ8P",
    "severity_val":   "5-ZVMT",
    "severity_test":  "5-ZWR2",
}

CATALOG_HOST  = "www.eye-ai.org"
CATALOG_ALIAS = "eye-ai"

# ---------------------------------------------------------------------------
# Label vocabulary maps
# ---------------------------------------------------------------------------

SEVERITY_MAP: dict[str, int] = {
    "Normal or No dx": 0,
    "GS":              1,
    "Mild":            2,
    "Moderate":        3,
    "Severe":          4,
}

BINARY_MAP: dict[str, int] = {
    "No Glaucoma":        0,
    "Suspected Glaucoma": 1,
}


# ---------------------------------------------------------------------------
# Connection & bag download
# ---------------------------------------------------------------------------

def connect_catalog(cache_dir=None):
    from eye_ai import EyeAI
    ml = EyeAI(hostname=CATALOG_HOST, catalog_id=CATALOG_ALIAS, cache_dir=cache_dir)
    logger.info("Connected to %s / %s", CATALOG_HOST, CATALOG_ALIAS)
    return ml


def download_bag(ml, dataset_rid: str) -> Path:
    ds   = ml.lookup_dataset(dataset_rid)
    info = ds.cache(ds.current_version)
    bag_root = Path(info["cache_path"])
    logger.info("Bag root: %s", bag_root)
    return bag_root


# ---------------------------------------------------------------------------
# Image paths
# ---------------------------------------------------------------------------

def find_image_csv(bag_root: Path) -> Path | None:
    candidates = list(bag_root.rglob("Image.csv"))
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def extract_image_paths(bag_root: Path) -> pd.DataFrame:
    asset_dir = bag_root / "data" / "asset"
    rows = []
    if asset_dir.exists():
        for rid_dir in sorted(asset_dir.iterdir()):
            img_dir = rid_dir / "Image"
            if img_dir.exists():
                files = [f for f in img_dir.iterdir() if f.is_file()]
                if files:
                    rows.append({"image_rid": rid_dir.name,
                                 "image_path": str(files[0])})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["image_rid", "image_path"])
    logger.info("Image files found: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Binary labels — Initial Diagnosis only
# ---------------------------------------------------------------------------

def extract_binary_labels(bag_root: Path, image_rids: set[str]) -> pd.DataFrame:
    """
    Extract binary labels from Image_Diagnosis.csv using Initial Diagnosis ONLY.
    Expert_Consensus rows are ignored entirely.
    """
    candidates = list(bag_root.rglob("Image_Diagnosis.csv"))
    if not candidates:
        logger.debug("No Image_Diagnosis.csv in bag %s", bag_root.name)
        return pd.DataFrame(columns=["image_rid", "binary_label"])

    diag_df = pd.read_csv(max(candidates, key=lambda p: p.stat().st_size))

    # Keep Initial Diagnosis rows only
    diag_df = diag_df[diag_df["Diagnosis_Tag"] == "Initial Diagnosis"]

    df = diag_df[diag_df["Image"].isin(image_rids)].copy()
    if df.empty:
        return pd.DataFrame(columns=["image_rid", "binary_label"])

    df["binary_label"] = df["Diagnosis_Image"].map(BINARY_MAP)
    df = df.dropna(subset=["binary_label"])
    df = df.drop_duplicates(subset=["Image"], keep="first")

    result = df[["Image", "binary_label"]].rename(columns={"Image": "image_rid"})
    logger.info("Binary labels (Initial Diagnosis only): %d images labeled",
                len(result))
    return result


# ---------------------------------------------------------------------------
# Severity labels from clinical records chain
# ---------------------------------------------------------------------------

def extract_severity_labels(bag_root: Path, image_csv: Path) -> pd.DataFrame:
    """
    Join Image → Observation → Clinical_Records_Observation
              → Clinical_Records → Execution_Clinical_Records_Glaucoma_Severity.

    Laterality: prefer rows where Image.Image_Side matches
    Clinical_Records.Powerform_Laterality.
    Multiple visits per image: keep most recent Date_of_Encounter.
    Returns severity grades 1–4 only (grade-0 is handled via binary_label).
    """
    base = image_csv.parent

    def read_rel(rel: str) -> pd.DataFrame:
        p = base / rel
        if not p.exists():
            logger.warning("Missing CSV: %s", p.relative_to(bag_root))
            return pd.DataFrame()
        return pd.read_csv(p)

    image_df = pd.read_csv(image_csv)[["RID", "Observation", "Image_Side"]]
    obs_df   = read_rel("Image/Observation.csv")
    cro_df   = read_rel("Image/Observation/Clinical_Records_Observation.csv")
    cr_df    = read_rel("Image/Observation/Clinical_Records_Observation/Clinical_Records.csv")
    sev_csv  = ("Image/Observation/Clinical_Records_Observation/Clinical_Records/"
                "Execution_Clinical_Records_Glaucoma_Severity.csv")
    sev_df   = read_rel(sev_csv)

    for name, df in [("Observation", obs_df), ("CRO", cro_df),
                     ("ClinicalRecords", cr_df), ("Severity", sev_df)]:
        if df.empty:
            logger.warning("Empty or missing table %s — no severity labels", name)
            return pd.DataFrame(columns=["image_rid", "severity_label"])

    df = image_df.rename(columns={"RID": "RID_img", "Observation": "Obs_fk"})
    df = df.merge(obs_df[["RID"]].rename(columns={"RID": "RID_obs"}),
                  left_on="Obs_fk", right_on="RID_obs")
    df = df.merge(cro_df[["Observation", "Clinical_Records"]].rename(
                      columns={"Observation": "Obs_fk2", "Clinical_Records": "CR_fk"}),
                  left_on="RID_obs", right_on="Obs_fk2")
    cr_keep = [c for c in ["RID", "Date_of_Encounter", "Powerform_Laterality"]
               if c in cr_df.columns]
    df = df.merge(cr_df[cr_keep].rename(columns={"RID": "CR_rid"}),
                  left_on="CR_fk", right_on="CR_rid")
    df = df.merge(sev_df[["Clinical_Records", "ICD_Severity_Label"]].rename(
                      columns={"Clinical_Records": "Sev_CR_fk"}),
                  left_on="CR_rid", right_on="Sev_CR_fk")

    df["severity_label"] = df["ICD_Severity_Label"].map(SEVERITY_MAP)
    df = df.dropna(subset=["severity_label"])

    # Exclude grade-0 from severity column (handled via binary_label)
    df = df[df["severity_label"] > 0]

    if "Image_Side" in df.columns and "Powerform_Laterality" in df.columns:
        mask_match   = df["Image_Side"].str.lower() == df["Powerform_Laterality"].str.lower()
        mask_unknown = df["Image_Side"].isna() | df["Powerform_Laterality"].isna()
        matched = df[mask_match | mask_unknown]
        if not matched.empty:
            df = matched

    if "Date_of_Encounter" in df.columns:
        df = df.sort_values("Date_of_Encounter", ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["RID_img"], keep="first")

    result = df[["RID_img", "severity_label"]].rename(
        columns={"RID_img": "image_rid"}).reset_index(drop=True)
    logger.info("Severity labels (grades 1–4): %d images labeled", len(result))
    return result


# ---------------------------------------------------------------------------
# Per-split manifest builder
# ---------------------------------------------------------------------------

def build_manifest_for_split(
    ml,
    dataset_rid: str,
    split: str,
    fetch_severity: bool,
) -> pd.DataFrame:
    logger.info("=== %s  dataset=%s ===", split.upper(), dataset_rid)

    bag_root  = download_bag(ml, dataset_rid)
    image_csv = find_image_csv(bag_root)

    if image_csv is None:
        logger.error("No Image.csv found in bag %s", bag_root)
        return pd.DataFrame(columns=["image_rid", "image_path",
                                     "binary_label", "severity_label", "split"])

    image_rids  = set(pd.read_csv(image_csv)["RID"].tolist())
    path_df     = extract_image_paths(bag_root)
    binary_df   = extract_binary_labels(bag_root, image_rids)
    severity_df = (extract_severity_labels(bag_root, image_csv)
                   if fetch_severity
                   else pd.DataFrame(columns=["image_rid", "severity_label"]))

    manifest = (path_df
                .merge(binary_df,   on="image_rid", how="left")
                .merge(severity_df, on="image_rid", how="left"))
    manifest["split"] = split

    _log_stats(manifest, split)
    return manifest


def _log_stats(df: pd.DataFrame, split: str) -> None:
    n  = len(df)
    nb = df["binary_label"].notna().sum()
    ns = df["severity_label"].notna().sum()
    logger.info(
        "  %s: %d images | binary=%d (%.0f%%) | severity=%d (%.0f%%)",
        split, n,
        nb, 100 * nb / n if n else 0,
        ns, 100 * ns / n if n else 0,
    )
    if ns > 0:
        vc = df["severity_label"].value_counts().sort_index()
        logger.info("  Severity distribution:\n%s", vc.to_string())


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> Path:
    output_dir    = Path(args.output_dir).resolve()
    manifest_path = output_dir / "manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    ml = connect_catalog(cache_dir=output_dir)

    #                          split              fetch_severity
    configs = [
        ("binary_train",   False),
        ("binary_val",     False),
        ("severity_train", True),
        ("severity_val",   True),
        ("severity_test",  True),
    ]

    parts = []
    for split, with_severity in configs:
        parts.append(build_manifest_for_split(
            ml, STUDY_DATASETS[split], split, with_severity))

    manifest = pd.concat(parts, ignore_index=True)

    # ── Combined ground-truth severity label ──────────────────────────────────
    # severity_label (1–4) takes precedence; binary_label==0 fills in grade-0.
    manifest["label"] = manifest["severity_label"].copy()
    grade0_mask = manifest["label"].isna() & (manifest["binary_label"] == 0)
    manifest.loc[grade0_mask, "label"] = 0

    # Drop rows with neither binary_label nor severity_label
    no_label = manifest["binary_label"].isna() & manifest["severity_label"].isna()
    n_dropped = int(no_label.sum())
    if n_dropped:
        logger.warning("Dropping %d images with no binary or severity label", n_dropped)
    manifest = manifest[~no_label].reset_index(drop=True)

    # Cast to nullable integer so NaN is preserved correctly in CSV
    manifest["binary_label"]   = manifest["binary_label"].astype("Int64")
    manifest["severity_label"] = manifest["severity_label"].astype("Int64")
    manifest["label"]          = manifest["label"].astype("Int64")

    manifest.to_csv(manifest_path, index=False)
    logger.info("Manifest saved → %s  (%d rows)", manifest_path, len(manifest))
    _log_final(manifest)
    return manifest_path


def _log_final(df: pd.DataFrame) -> None:
    logger.info("Final manifest summary:")
    for split, grp in df.groupby("split"):
        n  = len(grp)
        nb = grp["binary_label"].notna().sum()
        ns = grp["severity_label"].notna().sum()
        nl = grp["label"].notna().sum()
        logger.info("  %-20s %4d rows | binary=%d | severity=%d | label=%d",
                    split, n, nb, ns, nl)
    if "label" in df.columns:
        vc = df["label"].value_counts().sort_index()
        logger.info("  label distribution:\n%s", vc.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download v2 study datasets and build manifest.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="data/")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    manifest_path = run_pipeline(args)
    print(f"\nDone. Manifest: {manifest_path}")
