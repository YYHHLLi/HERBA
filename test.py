#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified HERBA checkpoint evaluator.

Expected release layout
-----------------------
HERBA/
├── Checkpoint/
│   ├── blind_double/
│   ├── blind_RNA/
│   ├── blind_small_molecule/
│   ├── classification_cv_10/
│   ├── classification_cv_5/
│   ├── molecular_scaffold/
│   ├── regression_cv_10/
│   ├── regression_cv_5/
│   └── rna_similarity_30/
│       ├── HERBA_f0.pt
│       ├── HERBA_f1.pt
│       ├── ...
│       └── dataset_cache/
│           ├── f0_test.data
│           ├── f1_test.data
│           └── ...
├── models/
│   └── HERBA.py
├── utils.py
└── test.py

Examples
--------
Regression:
    python test.py --task regression_cv_5 --gpu 0

Classification:
    python test.py --task classification_cv_5 --gpu 0

Cold-start:
    python test.py --task blind_RNA --gpu 0

A custom task directory can also be supplied:
    python test.py --task-dir /path/to/task_bundle --task-type regression --gpu 0
"""

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import torch

from models.HERBA import HERBA
from utils import (
    collate,
    predicting,
    rmse,
    mse,
    pearson,
    spearman,
    ci,
    rm2,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


DEFAULT_TASKS = {
    "regression_cv_5": {
        "type": "regression",
        "folds_per_repeat": 5,
    },
    "regression_cv_10": {
        "type": "regression",
        "folds_per_repeat": 10,
    },
    "classification_cv_5": {
        "type": "classification",
        "folds_per_repeat": 5,
    },
    "classification_cv_10": {
        "type": "classification",
        "folds_per_repeat": 10,
    },
    "blind_RNA": {
        "type": "regression",
        "folds_per_repeat": 5,
    },
    "blind_small_molecule": {
        "type": "regression",
        "folds_per_repeat": 5,
    },
    "blind_double": {
        "type": "regression",
        "folds_per_repeat": 5,
    },
    "rna_similarity_30": {
        "type": "regression",
        "folds_per_repeat": 5,
    },
    "molecular_scaffold": {
        "type": "regression",
        "folds_per_repeat": 5,
    },
}


def load_torch(path, map_location=None):

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def natural_model_index(path: Path) -> int:

    m = re.fullmatch(r"HERBA_f(\d+)\.pt", path.name)
    if m is None:
        raise ValueError(
            f"Unexpected checkpoint filename: {path.name}. "
            "Expected HERBA_f<number>.pt"
        )
    return int(m.group(1))


def discover_checkpoints(task_dir: Path):

    paths = list(task_dir.glob("HERBA_f*.pt"))
    paths = sorted(paths, key=natural_model_index)

    if not paths:
        raise FileNotFoundError(
            f"No checkpoint found in {task_dir}\n"
            "Expected files such as HERBA_f0.pt, HERBA_f1.pt, ..."
        )

    indices = [natural_model_index(p) for p in paths]
    if len(indices) != len(set(indices)):
        raise RuntimeError(
            f"Duplicate HERBA checkpoint indices found in {task_dir}"
        )

    return paths


def extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif checkpoint and all(
            isinstance(k, str) for k in checkpoint.keys()
        ) and any(
            torch.is_tensor(v) for v in checkpoint.values()
        ):
            state = checkpoint
        else:
            raise KeyError(
                "Cannot find model weights in checkpoint. "
                "Expected 'model_state_dict', 'state_dict', "
                "or a raw state_dict."
            )
    else:
        raise TypeError(
            "Unsupported checkpoint format. "
            "Please save HERBA weights as a state_dict or checkpoint dict."
        )


    if any(k.startswith("module.") for k in state):
        state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

    return state


def checkpoint_epoch(checkpoint):
    if not isinstance(checkpoint, dict):
        return ""
    for key in ("epoch", "best_epoch"):
        if key in checkpoint:
            try:
                return int(checkpoint[key])
            except Exception:
                return checkpoint[key]
    return ""


@torch.no_grad()
def predict_classification(model, device, loader):

    model.eval()

    y_true_all = []
    prob_all = []

    for drug, target in loader:
        drug = drug.to(device)
        target = target.to(device)

        logits = model(drug, target)
        probs = torch.sigmoid(logits)

        y_true_all.append(
            drug.y.view(-1).detach().cpu()
        )
        prob_all.append(
            probs.view(-1).detach().cpu()
        )

    if not y_true_all:
        raise RuntimeError("The test loader is empty.")

    return (
        torch.cat(y_true_all).numpy(),
        torch.cat(prob_all).numpy(),
    )


def classification_metrics(
    y_true,
    prob,
    probability_threshold=0.5,
):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    y_pred = (
        prob >= float(probability_threshold)
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )
    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = float("nan")

    try:
        auprc = average_precision_score(
            y_true,
            prob,
        )
    except ValueError:
        auprc = float("nan")

    return {
        "accuracy": accuracy_score(
            y_true, y_pred
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "bacc": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "mcc": matthews_corrcoef(
            y_true,
            y_pred,
        ),
        "auc": auc,
        "auprc": auprc,
    }


def mean_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan"), float("nan")

    mean = float(np.mean(values))
    std = (
        float(np.std(values, ddof=1))
        if len(values) > 1
        else 0.0
    )
    return mean, std


def print_summary(rows, metric_names):
    print("\n" + "=" * 80)
    print("HERBA TEST SUMMARY")
    print("=" * 80)

    for metric in metric_names:
        values = [
            float(row[metric])
            for row in rows
            if row.get(metric, "") != ""
        ]
        mean, std = mean_std(values)
        print(
            f"{metric:12s}: "
            f"{mean:.6f} ± {std:.6f}"
        )

    print("=" * 80)


def infer_task_type(task_name, explicit_type=None):
    if explicit_type is not None:
        return explicit_type

    if (
        task_name is not None
        and task_name in DEFAULT_TASKS
    ):
        return DEFAULT_TASKS[task_name]["type"]

    if task_name and "classification" in task_name.lower():
        return "classification"

    return "regression"


def infer_folds_per_repeat(
    task_name,
    explicit_folds=None,
):
    if explicit_folds is not None:
        return explicit_folds

    if (
        task_name is not None
        and task_name in DEFAULT_TASKS
    ):
        return DEFAULT_TASKS[
            task_name
        ]["folds_per_repeat"]

    if task_name and "_10" in task_name:
        return 10

    return 5


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate released HERBA checkpoints "
            "using the cached test partitions."
        )
    )

    parser.add_argument(
        "--task",
        choices=list(DEFAULT_TASKS.keys()),
        default=None,
        help=(
            "Released task name under Checkpoint/. "
            "Example: regression_cv_5 or blind_RNA."
        ),
    )

    parser.add_argument(
        "--task-dir",
        type=str,
        default=None,
        help=(
            "Optional custom task-bundle directory. "
            "If supplied, this overrides "
            "--checkpoint-root/--task."
        ),
    )

    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="Checkpoint",
        help="Root directory containing released task folders.",
    )

    parser.add_argument(
        "--task-type",
        choices=["regression", "classification"],
        default=None,
        help=(
            "Optional override for a custom --task-dir."
        ),
    )

    parser.add_argument(
        "--folds-per-repeat",
        type=int,
        choices=[5, 10],
        default=None,
        help=(
            "Used only to report rep/fold IDs from flattened "
            "HERBA_f*.pt indices."
        ),
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--rnafm-dim",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.5,
        help=(
            "Probability cutoff for classification predictions."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output CSV path. Default: "
            "<task_dir>/predict_result.csv"
        ),
    )

    args = parser.parse_args()

    if args.task_dir is None and args.task is None:
        parser.error(
            "Specify either --task or --task-dir."
        )

    if args.task_dir is not None:
        task_dir = Path(
            args.task_dir
        ).expanduser().resolve()
        task_name = (
            args.task
            if args.task is not None
            else task_dir.name
        )
    else:
        task_name = args.task
        task_dir = (
            Path(args.checkpoint_root)
            / task_name
        ).expanduser().resolve()

    if not task_dir.is_dir():
        raise FileNotFoundError(
            f"Task directory not found: {task_dir}"
        )

    cache_dir = task_dir / "dataset_cache"
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"dataset_cache directory not found: "
            f"{cache_dir}"
        )

    task_type = infer_task_type(
        task_name,
        args.task_type,
    )
    folds_per_repeat = infer_folds_per_repeat(
        task_name,
        args.folds_per_repeat,
    )

    device = torch.device(
        f"cuda:{args.gpu}"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoints = discover_checkpoints(
        task_dir
    )

    print("=" * 80)
    print("HERBA checkpoint evaluation")
    print(f"Task       : {task_name}")
    print(f"Type       : {task_type}")
    print(f"Task dir   : {task_dir}")
    print(f"Cache dir  : {cache_dir}")
    print(f"Device     : {device}")
    print(
        f"Checkpoints: {len(checkpoints)}"
    )
    print("=" * 80)

    rows = []

    for checkpoint_path in checkpoints:
        flat_index = natural_model_index(
            checkpoint_path
        )

        test_path = (
            cache_dir
            / f"f{flat_index}_test.data"
        )

        if not test_path.is_file():
            raise FileNotFoundError(
                "Checkpoint/cache mismatch:\n"
                f"  checkpoint: {checkpoint_path}\n"
                f"  missing   : {test_path}"
            )

        rep = flat_index // folds_per_repeat
        fold = flat_index % folds_per_repeat

        print(
            f"\n[{flat_index}] "
            f"rep={rep} fold={fold} "
            f"checkpoint={checkpoint_path.name}"
        )
        print(
            f"     test={test_path.name}"
        )

        test_data = load_torch(
            test_path,
            map_location="cpu",
        )

        loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
        )

        model = HERBA(
            rnafm_dim=args.rnafm_dim
        ).to(device)

        checkpoint = load_torch(
            checkpoint_path,
            map_location=device,
        )
        state_dict = extract_state_dict(
            checkpoint
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        best_epoch = checkpoint_epoch(
            checkpoint
        )

        if task_type == "classification":
            y_true, prob = (
                predict_classification(
                    model,
                    device,
                    loader,
                )
            )

            metrics = classification_metrics(
                y_true,
                prob,
                probability_threshold=(
                    args.prob_threshold
                ),
            )

            row = {
                "task": task_name,
                "flat_fold": flat_index,
                "rep": rep,
                "fold": fold,
                "best_epoch": best_epoch,
                **{
                    k: float(v)
                    for k, v in metrics.items()
                },
            }

            print(
                "     "
                f"BACC={metrics['bacc']:.6f} "
                f"AUC={metrics['auc']:.6f} "
                f"Precision={metrics['precision']:.6f} "
                f"Specificity={metrics['specificity']:.6f}"
            )

        else:
            y_true, y_pred = predicting(
                model,
                device,
                loader,
            )

            metrics = {
                "rmse": rmse(
                    y_true, y_pred
                ),
                "mse": mse(
                    y_true, y_pred
                ),
                "pearson": pearson(
                    y_true, y_pred
                ),
                "spearman": spearman(
                    y_true, y_pred
                ),
                "ci": ci(
                    y_true, y_pred
                ),
                "rm2": rm2(
                    y_true, y_pred
                ),
            }

            row = {
                "task": task_name,
                "flat_fold": flat_index,
                "rep": rep,
                "fold": fold,
                "best_epoch": best_epoch,
                **{
                    k: float(v)
                    for k, v in metrics.items()
                },
            }

            print(
                "     "
                f"RMSE={metrics['rmse']:.6f} "
                f"PCC={metrics['pearson']:.6f} "
                f"SCC={metrics['spearman']:.6f}"
            )

        rows.append(row)

        del model, test_data, loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError(
            "No checkpoint was evaluated."
        )

    output_path = (
        Path(args.output)
        if args.output is not None
        else task_dir / "predict_result.csv"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with open(
        output_path,
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    if task_type == "classification":
        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "bacc",
            "mcc",
            "auc",
            "auprc",
        ]
    else:
        metric_names = [
            "rmse",
            "mse",
            "pearson",
            "spearman",
            "ci",
            "rm2",
        ]

    print_summary(
        rows,
        metric_names,
    )

    print(
        f"\nPer-fold results saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
