"""
construct_datasets_v2.py

Creates 6 study datasets in the eye-ai catalog:

  1. Glaucoma_Triage_Binary_Train_v2   — angle-2 from LAC train (2-277G),
                                          one eye per subject (severer Initial Diagnosis label)
  2. Glaucoma_Triage_Binary_Val_v2     — angle-2 from LAC val (2-277J), same
  3. Glaucoma_Triage_Binary_Test_v2    — angle-2 from LAC test (2-277C), same
  4. Glaucoma_Triage_Severity_Train_v2 — 80% of 4-4116 (seed=42) + 1:1 grade-0 from binary_train
  5. Glaucoma_Triage_Severity_Val_v2   — 20% of 4-4116 (seed=42) + 1:1 grade-0 from binary_val
  6. Glaucoma_Triage_Severity_Test_v2  — all of 4-411G + 1:1 grade-0 from binary_test

Key differences from construct_datasets.py (v1):
  - "Initial Diagnosis" only — Expert_Consensus is never used or prioritized
  - Images with no Initial Diagnosis label are dropped from binary datasets
  - New binary_test dataset (v1 had no binary_test)
  - Severity train/val is 80/20 (v1 was 85/15)
  - Grade-0 images come from the corresponding binary split (already one-per-subject)
  - 1:1 cap applied to all three severity splits (v1 only capped the test split)

Usage
-----
    # Dry-run — print counts, no catalog writes
    python src/catalog/construct_datasets_v2.py --dry-run

    # Create all 6 datasets
    python src/catalog/construct_datasets_v2.py
"""
from __future__ import annotations

import argparse
import logging
import random

logger = logging.getLogger(__name__)

DEFAULTS = {
    "lac_train": "2-277G",
    "lac_val":   "2-277J",
    "lac_test":  "2-277C",
    "sev_train": "4-4116",
    "sev_test":  "4-411G",
}

CATALOG_HOST  = "www.eye-ai.org"
CATALOG_ALIAS = "eye-ai"

BINARY_MAP: dict[str, int] = {
    "No Glaucoma":        0,
    "Suspected Glaucoma": 1,
}

WORKFLOW_NAME = "Glaucoma Triage Dataset Construction v2"
WORKFLOW_URL  = "https://github.com/Zhiweiii/Glaucoma-Triage-DFL/blob/main/src/construct_datasets_v2.py"

SEV_VAL_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect_catalog():
    from eye_ai import EyeAI
    ml = EyeAI(hostname=CATALOG_HOST, catalog_id=CATALOG_ALIAS)
    logger.info("Connected to %s / %s", CATALOG_HOST, CATALOG_ALIAS)
    return ml


# ---------------------------------------------------------------------------
# pathBuilder helpers
# ---------------------------------------------------------------------------

def _get_pb_tables(ml):
    pb = ml.pathBuilder()
    s = pb.schemas["eye-ai"].tables
    return {
        "SubjectDataset": s["Subject_Dataset"],
        "Subject":        s["Subject"],
        "Observation":    s["Observation"],
        "Image":          s["Image"],
        "ImageDiagnosis": s["Image_Diagnosis"],
        "DatasetImage":   s["Dataset_Image"],
    }


def get_rids_from_image_dataset(ml, dataset_rid: str) -> list[str]:
    """All image RIDs from an image-based dataset."""
    t = _get_pb_tables(ml)
    DatasetImage = t["DatasetImage"]
    Image        = t["Image"]
    path = DatasetImage.filter(DatasetImage.columns["Dataset"] == dataset_rid).link(Image)
    rids = [r["RID"] for r in path.attributes(path.Image.RID)]
    logger.info("Dataset %s → %d images", dataset_rid, len(rids))
    return rids


def get_angle2_quads_initial_diag_only(
    ml, lac_rid: str,
) -> list[tuple[str, str, str, int]]:
    """
    (image_rid, subject_rid, image_side, binary_label) for all angle-2 images
    in a LAC (subject-based) dataset, using Initial Diagnosis labels ONLY.
    Images with no Initial Diagnosis label are excluded entirely.
    """
    t = _get_pb_tables(ml)
    SubjectDataset = t["SubjectDataset"]
    Subject        = t["Subject"]
    Observation    = t["Observation"]
    Image          = t["Image"]
    ImageDiagnosis = t["ImageDiagnosis"]

    # All angle-2 images with subject + side info
    base_path = (SubjectDataset
                 .filter(SubjectDataset.columns["Dataset"] == lac_rid)
                 .link(Subject)
                 .link(Observation)
                 .link(Image)
                 .filter(Image.columns["Image_Angle"] == "2"))
    base_rows = list(base_path.attributes(
        base_path.Image.RID,
        base_path.Image.Image_Side,
        base_path.Observation.Subject,
    ))
    info: dict[str, tuple[str, str]] = {
        r["RID"]: (str(r["Subject"]), r.get("Image_Side") or "")
        for r in base_rows
    }
    logger.info("LAC %s — %d angle-2 images total", lac_rid, len(info))

    # Initial Diagnosis labels only — Expert_Consensus rows are ignored
    label_path = (SubjectDataset
                  .filter(SubjectDataset.columns["Dataset"] == lac_rid)
                  .link(Subject)
                  .link(Observation)
                  .link(Image)
                  .filter(Image.columns["Image_Angle"] == "2")
                  .link(ImageDiagnosis)
                  .filter(ImageDiagnosis.columns["Diagnosis_Tag"] == "Initial Diagnosis"))
    label_rows = list(label_path.attributes(
        label_path.Image.RID,
        label_path.Image_Diagnosis.Diagnosis_Image,
    ))

    labels: dict[str, int] = {}
    for r in label_rows:
        rid   = r["RID"]
        label = BINARY_MAP.get(r.get("Diagnosis_Image", ""), -1)
        if label >= 0:
            labels[rid] = label

    n_dropped = sum(1 for rid in info if rid not in labels)
    if n_dropped:
        logger.warning("  %d images have no Initial Diagnosis label — dropped", n_dropped)

    result = [
        (rid, subj, side, labels[rid])
        for rid, (subj, side) in info.items()
        if rid in labels
    ]
    logger.info("  → %d images retained with Initial Diagnosis label", len(result))
    return result


# ---------------------------------------------------------------------------
# Per-subject selection
# ---------------------------------------------------------------------------

def _select_severer_per_subject(
    quads: list[tuple[str, str, str, int]],
) -> list[tuple[str, int]]:
    """
    One image per subject: prefer highest binary_label, then Right > Left, then RID.
    Returns list of (image_rid, label).
    """
    side_prio = {"Right": 0, "Left": 1}
    groups: dict[str, list] = {}
    for rid, subject, side, label in quads:
        groups.setdefault(subject, []).append((-label, side_prio.get(side, 2), rid, label))
    return [(sorted(imgs)[0][2], sorted(imgs)[0][3]) for imgs in groups.values()]


# ---------------------------------------------------------------------------
# Workflow helpers
# ---------------------------------------------------------------------------

def get_or_create_workflow(ml):
    try:
        wf = ml.lookup_workflow_by_url(WORKFLOW_URL)
        logger.info("Reusing existing workflow %s", wf.rid if hasattr(wf, "rid") else wf)
        return wf
    except Exception:
        pass
    workflow = ml.create_workflow(
        name=WORKFLOW_NAME,
        workflow_type="Dataset_Management",
        description=(
            "Creates 6 study datasets for glaucoma triage v2: "
            "binary train/val/test (one eye per subject, Initial Diagnosis only — "
            "Expert_Consensus not used), severity train/val (80/20 split of 4-4116 + "
            "1:1 grade-0 from corresponding binary split), severity test "
            "(4-411G + 1:1 grade-0 from binary_test)."
        ),
    )
    rid = ml.add_workflow(workflow)
    logger.info("Created workflow %s", rid)
    return ml.lookup_workflow(rid)


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_binary_dataset(
    ml, exe,
    name: str, description: str, dataset_type: str,
    lac_rid: str, dry_run: bool,
) -> tuple[list[str], list[str]]:
    """
    Builds a binary dataset from a LAC source.
    Returns (all_keep_rids, grade0_rids) where grade0_rids is the subset
    with label=0 (No Glaucoma) — used as the grade-0 pool for severity datasets.
    """
    quads = get_angle2_quads_initial_diag_only(ml, lac_rid)
    pairs = _select_severer_per_subject(quads)   # [(rid, label), ...]

    keep_rids   = [rid   for rid, _     in pairs]
    grade0_rids = [rid   for rid, label in pairs if label == 0]

    logger.info("  %s: %d images (one/subject) | %d No-Glaucoma (grade-0)",
                name, len(keep_rids), len(grade0_rids))

    if not dry_run:
        ds = exe.create_dataset(
            dataset_types=[dataset_type, "Image"],
            description=description,
        )
        ds.add_dataset_members(
            members={"Image": keep_rids},
            description=(
                f"Angle-2 images from LAC {lac_rid}, one per subject "
                "(severer Initial Diagnosis label, prefer Right eye). "
                "Expert_Consensus labels not used."
            ),
        )
        logger.info("  Created %s → %s (%d images)", name, ds.dataset_rid, len(keep_rids))

    return keep_rids, grade0_rids


def build_severity_dataset(
    ml, exe,
    name: str, description: str, dataset_type: str,
    sev_rids: list[str], grade0_pool: list[str],
    seed: int, dry_run: bool,
) -> None:
    """
    Severity images + grade-0 from the corresponding binary split, capped 1:1.
    grade0_pool is already one-per-subject (sourced from the binary dataset).
    """
    n_sev  = len(sev_rids)
    grade0 = list(grade0_pool)
    if len(grade0) > n_sev:
        rng    = random.Random(seed)
        grade0 = rng.sample(grade0, n_sev)
        logger.info("  Capped grade-0 to %d (1:1 with severity images)", n_sev)

    all_rids = sev_rids + grade0
    logger.info("  %s: %d severity + %d grade-0 = %d total",
                name, n_sev, len(grade0), len(all_rids))

    if dry_run:
        return

    ds = exe.create_dataset(
        dataset_types=[dataset_type, "Image"],
        description=description,
    )
    ds.add_dataset_members(
        members={"Image": all_rids},
        description=description,
    )
    logger.info("  Created %s → %s (%d images)", name, ds.dataset_rid, len(all_rids))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    ml = connect_catalog()

    if args.dry_run:
        logger.info("DRY RUN — no catalog writes")

    workflow = None if args.dry_run else get_or_create_workflow(ml)

    # ── Split severity source images 80/20 ───────────────────────────────────
    all_sev_rids = get_rids_from_image_dataset(ml, args.sev_train)
    rng      = random.Random(args.seed)
    shuffled = list(all_sev_rids)
    rng.shuffle(shuffled)
    n_val          = max(1, int(len(shuffled) * SEV_VAL_FRACTION))
    sev_val_rids   = shuffled[:n_val]
    sev_train_rids = shuffled[n_val:]
    logger.info("4-4116 split (seed=%d): %d train (80%%) + %d val (20%%)",
                args.seed, len(sev_train_rids), len(sev_val_rids))

    sev_test_rids = get_rids_from_image_dataset(ml, args.sev_test)

    from contextlib import nullcontext
    exe_ctx = (
        nullcontext(None) if args.dry_run
        else ml.create_execution(
            workflow=workflow,
            description="Construct 6 study datasets for glaucoma triage v2 (Initial Diagnosis only)",
        )
    )
    with exe_ctx as exe:

        # ── Binary datasets ───────────────────────────────────────────────────
        _, grade0_train = build_binary_dataset(
            ml, exe,
            name="Glaucoma_Triage_Binary_Train_v2",
            description=(
                "Angle-2 fundus images for subjects in LAC train (2-277G), "
                "one eye per subject (severer Initial Diagnosis label, prefer Right eye). "
                "Initial Diagnosis only — Expert_Consensus not used."
            ),
            dataset_type="Training",
            lac_rid=args.lac_train,
            dry_run=args.dry_run,
        )

        _, grade0_val = build_binary_dataset(
            ml, exe,
            name="Glaucoma_Triage_Binary_Val_v2",
            description=(
                "Angle-2 fundus images for subjects in LAC val (2-277J), "
                "one eye per subject (severer Initial Diagnosis label, prefer Right eye). "
                "Initial Diagnosis only — Expert_Consensus not used."
            ),
            dataset_type="Validation",
            lac_rid=args.lac_val,
            dry_run=args.dry_run,
        )

        _, grade0_test = build_binary_dataset(
            ml, exe,
            name="Glaucoma_Triage_Binary_Test_v2",
            description=(
                "Angle-2 fundus images for subjects in LAC test (2-277C), "
                "one eye per subject (severer Initial Diagnosis label, prefer Right eye). "
                "Initial Diagnosis only — Expert_Consensus not used."
            ),
            dataset_type="Test",
            lac_rid=args.lac_test,
            dry_run=args.dry_run,
        )

        # ── Severity datasets ─────────────────────────────────────────────────
        build_severity_dataset(
            ml, exe,
            name="Glaucoma_Triage_Severity_Train_v2",
            description=(
                f"80% of severity images from 4-4116 (seed={args.seed}) + "
                "1:1 grade-0 (No Glaucoma, Initial Diagnosis) from binary_train. "
                "Used for severity-head and DFL fine-tuning."
            ),
            dataset_type="Training",
            sev_rids=sev_train_rids,
            grade0_pool=grade0_train,
            seed=args.seed,
            dry_run=args.dry_run,
        )

        build_severity_dataset(
            ml, exe,
            name="Glaucoma_Triage_Severity_Val_v2",
            description=(
                f"20% hold-out of severity images from 4-4116 (seed={args.seed}) + "
                "1:1 grade-0 (No Glaucoma, Initial Diagnosis) from binary_val. "
                "Used for early stopping during severity and DFL training."
            ),
            dataset_type="Validation",
            sev_rids=sev_val_rids,
            grade0_pool=grade0_val,
            seed=args.seed,
            dry_run=args.dry_run,
        )

        build_severity_dataset(
            ml, exe,
            name="Glaucoma_Triage_Severity_Test_v2",
            description=(
                "All severity images from 4-411G + "
                "1:1 grade-0 (No Glaucoma, Initial Diagnosis) from binary_test. "
                "Represents a realistic screening mix for scheduling-cost evaluation."
            ),
            dataset_type="Test",
            sev_rids=sev_test_rids,
            grade0_pool=grade0_test,
            seed=args.seed,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        logger.info("Done. Update STUDY_DATASETS in data_pipeline.py with the new RIDs above.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create 6 glaucoma triage study datasets v2 (Initial Diagnosis only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print image counts only; do not write to catalog")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for 80/20 severity split and grade-0 capping")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--lac-train", default=DEFAULTS["lac_train"])
    p.add_argument("--lac-val",   default=DEFAULTS["lac_val"])
    p.add_argument("--lac-test",  default=DEFAULTS["lac_test"])
    p.add_argument("--sev-train", default=DEFAULTS["sev_train"])
    p.add_argument("--sev-test",  default=DEFAULTS["sev_test"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args)
