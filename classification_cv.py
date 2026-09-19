#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HERBA classification cross-validation.

This script supports both 5-fold and 10-fold binary classification
experiments using pKd >= threshold as the positive class.

Released directory layout
-------------------------
HERBA/
├── Checkpoint/
│   ├── classification_cv_5/
│   │   ├── HERBA_f0.pt
│   │   ├── HERBA_f1.pt
│   │   ├── ...
│   │   ├── dataset_cache/
│   │   │   ├── f0_train.data
│   │   │   ├── f0_val.data
│   │   │   ├── f0_test.data
│   │   │   └── ...
│   │   └── outer_test_results.csv
│   └── classification_cv_10/
│       └── ...
├── data/
├── models/
│   └── HERBA.py
├── create_data_fold.py
├── kmer.py
├── utils.py
└── classification_cv.py

Examples
--------
Train 5-fold classification:
    python classification_cv.py --action train --folds 5 --gpu 0

Test released 5-fold checkpoints:
    python classification_cv.py --action test --folds 5 --gpu 0

Train 10-fold classification:
    python classification_cv.py --action train --folds 10 --gpu 0

Test released 10-fold checkpoints:
    python classification_cv.py --action test --folds 10 --gpu 0
"""

import argparse
import csv
import json
import os
import pickle
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold, train_test_split
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

from create_data_fold import (
    smile_to_graph,
    label_smiles,
    clean_rna_seq,
    seq_cat,
    target_to_graph,
)
from kmer import (
    seq_to_kmer_freq,
    fit_kmer_zscore,
    load_kmer_zscore,
    zscore,
)
from models.HERBA import HERBA
from utils import TestbedDataset, collate


MODEL_NAME = "HERBA"
SPLIT_VERSION = "classification_nested_v1"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def atomic_save(payload, path):
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json_dump(obj, path):
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def upsert_csv(path, row, key_cols, columns):
    path = str(path)
    previous = []

    if os.path.isfile(path):
        with open(path, "r", newline="") as f:
            previous = list(csv.DictReader(f))

    key = tuple(str(row[k]) for k in key_cols)
    kept = []

    for old in previous:
        old_key = tuple(str(old[k]) for k in key_cols)
        if old_key != key:
            kept.append(old)

    kept.append({c: row[c] for c in columns})

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(kept)


def load_raw(data_root, dataset="RNA"):
    fpath = os.path.join(
        str(data_root),
        dataset,
    )

    with open(
        os.path.join(fpath, "ligand.json"),
        "r",
    ) as f:
        ligands = json.load(
            f,
            object_pairs_hook=OrderedDict,
        )

    with open(
        os.path.join(fpath, "rna.json"),
        "r",
    ) as f:
        proteins = json.load(
            f,
            object_pairs_hook=OrderedDict,
        )

    with open(
        os.path.join(fpath, "Y"),
        "rb",
    ) as f:
        affinity = pickle.load(
            f,
            encoding="latin1",
        )

    return (
        fpath,
        ligands,
        proteins,
        np.asarray(affinity),
    )


def build_feature_store(
    data_root,
    dataset,
    ligands,
    proteins,
    rnafm_dim,
):
    fpath = os.path.join(
        str(data_root),
        dataset,
    )

    aln_path = os.path.join(
        fpath,
        "aln",
    )
    pconsc_path = os.path.join(
        fpath,
        "pconsc4",
    )
    rnafm_path = os.path.join(
        fpath,
        "rnafm",
    )

    if not (
        os.path.exists(aln_path)
        and os.path.exists(pconsc_path)
    ):
        raise RuntimeError(
            "Missing aln or pconsc4 directory."
        )

    ligand_keys = list(ligands.keys())
    ligand_values = list(ligands.values())
    protein_keys = list(proteins.keys())

    drug_df = pd.read_csv(
        os.path.join(
            str(data_root),
            "drug_embeddings.csv",
        )
    )
    drug_df["ID"] = drug_df["ID"].astype(str)

    drug_emb_map = {
        row["ID"]: row.iloc[2:].values.astype(
            np.float32
        )
        for _, row in drug_df.iterrows()
    }

    smile_graph = {
        s: smile_to_graph(s)
        for s in ligand_values
    }

    smile_tensor = {
        s: label_smiles(s, 100)
        for s in ligand_values
    }

    seq_for_key = {
        k: clean_rna_seq(v)
        for k, v in proteins.items()
    }

    target_graph = {
        key: target_to_graph(
            key,
            seq_for_key[key],
            pconsc_path,
            aln_path,
            rnafm_dir=rnafm_path,
            rnafm_dim_default=rnafm_dim,
        )
        for key in protein_keys
    }

    return {
        "ligand_keys": ligand_keys,
        "ligand_values": ligand_values,
        "protein_keys": protein_keys,
        "drug_emb_map": drug_emb_map,
        "smile_graph": smile_graph,
        "smile_tensor": smile_tensor,
        "seq_for_key": seq_for_key,
        "target_graph": target_graph,
    }


def make_dataset_from_pairs(
    dataset_name,
    rows,
    cols,
    y_values,
    store,
    kmer_map,
    rnafm_dim,
):
    ligand_keys = store["ligand_keys"]
    ligand_values = store["ligand_values"]
    protein_keys = store["protein_keys"]

    keys = [
        protein_keys[int(c)]
        for c in cols
    ]

    return TestbedDataset(
        root="/tmp",
        dataset=dataset_name,
        xd=[
            ligand_values[int(r)]
            for r in rows
        ],
        xt=[
            seq_cat(
                store["seq_for_key"][
                    protein_keys[int(c)]
                ]
            )
            for c in cols
        ],
        y=np.asarray(
            y_values,
            dtype=np.float32,
        ),
        smile_graph=store["smile_graph"],
        smile_tensor=store["smile_tensor"],
        target_graph={
            k: store["target_graph"][k][:3]
            for k in keys
        },
        target_key=keys,
        kmer_map=kmer_map,
        rnafm_map={
            k: store["target_graph"][k][3]
            for k in keys
        },
        ligand_ids=np.asarray(
            ligand_keys
        )[np.asarray(rows, dtype=int)],
        drug_emb_map=store["drug_emb_map"],
    )


def fit_kmer_from_train_cols(
    data_root,
    dataset,
    train_cols,
    store,
    z_path,
):
    protein_keys = store["protein_keys"]

    unique_train_keys = sorted(
        {
            protein_keys[int(c)]
            for c in train_cols
        }
    )

    if len(unique_train_keys) == 0:
        raise RuntimeError(
            "No training RNA entities available "
            "for k-mer fitting."
        )

    if not os.path.exists(z_path):
        train_seqs = [
            store["seq_for_key"][k]
            for k in unique_train_keys
        ]
        fit_kmer_zscore(
            train_seqs,
            save_path=z_path,
        )

    mean, std = load_kmer_zscore(z_path)
    return mean, std


def make_kmer_map(
    cols_groups,
    store,
    mean,
    std,
):
    protein_keys = store["protein_keys"]

    all_cols = np.concatenate(
        [
            np.asarray(x, dtype=int)
            for x in cols_groups
            if len(x) > 0
        ]
    )

    needed = sorted(
        {
            protein_keys[int(c)]
            for c in all_cols
        }
    )

    return {
        str(k): zscore(
            seq_to_kmer_freq(
                store["seq_for_key"][k]
            ),
            mean,
            std,
        )
        for k in needed
    }


def binary_label(y, threshold=4.0):
    return (
        np.asarray(
            y,
            dtype=np.float32,
        )
        >= float(threshold)
    ).astype(np.float32)


def get_stratified_pair_folds(
    fpath,
    affinity,
    threshold,
    n_splits,
    rep,
):


    rows_all, cols_all = np.where(
        ~np.isnan(affinity)
    )

    y_all = binary_label(
        affinity[
            rows_all,
            cols_all,
        ],
        threshold,
    )

    fold_dir = os.path.join(
        fpath,
        "folds",
    )

    fold_file = os.path.join(
        fold_dir,
        (
            f"cls_pKd{threshold:g}_"
            f"stratified_{n_splits}_"
            f"rep{rep}.json"
        ),
    )

    if os.path.exists(fold_file):
        with open(
            fold_file,
            "r",
        ) as f:
            folds = [
                [int(x) for x in fold]
                for fold in json.load(f)
            ]

        if len(folds) != n_splits:
            raise RuntimeError(
                f"{fold_file} does not contain "
                f"{n_splits} folds."
            )

        return (
            rows_all,
            cols_all,
            y_all,
            folds,
        )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=1 + rep,
    )

    idx = np.arange(
        len(rows_all)
    )

    folds = [
        test_idx.tolist()
        for _, test_idx
        in skf.split(
            idx,
            y_all,
        )
    ]

    atomic_json_dump(
        folds,
        fold_file,
    )

    return (
        rows_all,
        cols_all,
        y_all,
        folds,
    )


def create_classification_dataset(
    data_root,
    dataset,
    rep,
    fold,
    n_splits,
    val_fraction,
    threshold,
    rnafm_dim,
):
    (
        fpath,
        ligands,
        proteins,
        affinity,
    ) = load_raw(
        data_root,
        dataset,
    )

    (
        rows_all,
        cols_all,
        y_all,
        folds,
    ) = get_stratified_pair_folds(
        fpath,
        affinity,
        threshold,
        n_splits,
        rep,
    )

    test_idx = np.asarray(
        folds[fold],
        dtype=int,
    )

    outer_train_idx = np.concatenate(
        [
            np.asarray(x, dtype=int)
            for i, x in enumerate(folds)
            if i != fold
        ]
    )

    train_idx, val_idx = train_test_split(
        outer_train_idx,
        test_size=val_fraction,
        random_state=(
            2026
            + rep * 1000
            + fold
        ),
        shuffle=True,
        stratify=y_all[
            outer_train_idx
        ],
    )

    tr_r = rows_all[train_idx]
    tr_c = cols_all[train_idx]

    va_r = rows_all[val_idx]
    va_c = cols_all[val_idx]

    te_r = rows_all[test_idx]
    te_c = cols_all[test_idx]

    tr_y = y_all[train_idx]
    va_y = y_all[val_idx]
    te_y = y_all[test_idx]

    print(
        f"[split] cls pKd>={threshold:g} "
        f"rep={rep} "
        f"fold={fold}/{n_splits}: "
        f"train={len(tr_y)} "
        f"pos={int(tr_y.sum())} "
        f"neg={int(len(tr_y)-tr_y.sum())}; "
        f"val={len(va_y)} "
        f"pos={int(va_y.sum())}; "
        f"test={len(te_y)} "
        f"pos={int(te_y.sum())}"
    )

    store = build_feature_store(
        data_root,
        dataset,
        ligands,
        proteins,
        rnafm_dim,
    )

    z_path = os.path.join(
        fpath,
        (
            f"kmer_z_cls_pKd{threshold:g}_"
            f"{n_splits}fold_"
            f"r{rep}_f{fold}.npz"
        ),
    )

    km_mean, km_std = (
        fit_kmer_from_train_cols(
            data_root,
            dataset,
            tr_c,
            store,
            z_path,
        )
    )

    kmer_map = make_kmer_map(
        [
            tr_c,
            va_c,
            te_c,
        ],
        store,
        km_mean,
        km_std,
    )

    train_data = make_dataset_from_pairs(
        (
            f"{dataset}_cls{n_splits}_"
            f"r{rep}_f{fold}_train"
        ),
        tr_r,
        tr_c,
        tr_y,
        store,
        kmer_map,
        rnafm_dim,
    )

    val_data = make_dataset_from_pairs(
        (
            f"{dataset}_cls{n_splits}_"
            f"r{rep}_f{fold}_val"
        ),
        va_r,
        va_c,
        va_y,
        store,
        kmer_map,
        rnafm_dim,
    )

    test_data = make_dataset_from_pairs(
        (
            f"{dataset}_cls{n_splits}_"
            f"r{rep}_f{fold}_test"
        ),
        te_r,
        te_c,
        te_y,
        store,
        kmer_map,
        rnafm_dim,
    )

    return (
        train_data,
        val_data,
        test_data,
    )


def cls_pos_weight(dataset, device):
    y = np.asarray(
        [
            float(
                d.y.view(-1)[0].item()
            )
            for d in dataset.DrugData
        ],
        dtype=np.float32,
    )

    pos = float(
        y.sum()
    )
    neg = float(
        len(y) - pos
    )

    if pos <= 0:
        return torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=device,
        )

    return torch.tensor(
        [neg / pos],
        dtype=torch.float32,
        device=device,
    )


def train_cls_epoch(
    model,
    device,
    loader,
    optimizer,
    criterion,
    epoch,
):
    model.train()

    running = 0.0
    n = 0

    for drug, target in loader:
        drug = drug.to(device)
        target = target.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            drug,
            target,
        )

        y = drug.y.view(
            -1,
            1,
        ).float()

        loss = criterion(
            logits,
            y,
        )

        loss.backward()
        optimizer.step()

        running += float(
            loss.item()
        )
        n += 1

    value = running / max(
        1,
        n,
    )

    print(
        f"epoch={epoch} "
        f"train_bce={value:.6f}"
    )

    return value


@torch.no_grad()
def predict_cls(
    model,
    device,
    loader,
):
    model.eval()

    ys = []
    probs = []

    for drug, target in loader:
        drug = drug.to(device)
        target = target.to(device)

        logits = model(
            drug,
            target,
        )

        prob = torch.sigmoid(
            logits
        )

        ys.append(
            drug.y.view(-1)
            .detach()
            .cpu()
        )

        probs.append(
            prob.view(-1)
            .detach()
            .cpu()
        )

    if not ys:
        raise RuntimeError(
            "The evaluation loader is empty."
        )

    return (
        torch.cat(ys).numpy(),
        torch.cat(probs).numpy(),
    )


def cls_metrics(
    y_true,
    prob,
    cutoff=0.5,
):
    y_true = np.asarray(
        y_true
    ).astype(int)

    prob = np.asarray(
        prob,
        dtype=float,
    )

    pred = (
        prob >= float(cutoff)
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    try:
        auc = roc_auc_score(
            y_true,
            prob,
        )
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
            y_true,
            pred,
        ),
        "precision": precision_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "bacc": balanced_accuracy_score(
            y_true,
            pred,
        ),
        "mcc": matthews_corrcoef(
            y_true,
            pred,
        ),
        "auc": auc,
        "auprc": auprc,
    }


def main():
    project_root = Path(
        __file__
    ).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "HERBA binary classification "
            "5-fold/10-fold cross-validation."
        )
    )

    parser.add_argument(
        "--action",
        choices=[
            "train",
            "test",
        ],
        default="train",
    )

    parser.add_argument(
        "--folds",
        type=int,
        choices=[
            5,
            10,
        ],
        default=5,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help=(
            "Positive class definition: "
            "pKd >= threshold."
        ),
    )

    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold for "
            "binary prediction."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--batch-test",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--rnafm-dim",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get(
            "HERBA_DATA_DIR",
            str(project_root / "data"),
        ),
        help=(
            "HERBA data directory. "
            "Default: ./data"
        ),
    )

    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=str(
            project_root
            / "Checkpoint"
        ),
        help=(
            "Root directory for released "
            "checkpoint bundles."
        ),
    )

    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help=(
            "Rebuild train/val/test .data files "
            "when training, even if they already exist."
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        (
            f"cuda:{args.gpu}"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    task_name = (
        f"classification_cv_{args.folds}"
    )

    task_dir = (
        Path(args.checkpoint_root)
        / task_name
    )

    cache_dir = (
        task_dir
        / "dataset_cache"
    )

    task_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        task_dir
        / "outer_test_results.csv"
    )

    columns = [
        "dataset",
        "rep",
        "fold",
        "flat_fold",
        "model",
        "best_epoch",
        "val_bacc",
        "val_loss",
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

    print("=" * 80)
    print("HERBA classification CV")
    print(
        f"Action          : {args.action}"
    )
    print(
        f"Folds           : {args.folds}"
    )
    print(
        f"Repeats         : {args.repeats}"
    )
    print(
        f"pKd threshold   : {args.threshold}"
    )
    print(
        f"Pred threshold  : {args.prob_threshold}"
    )
    print(
        f"Device          : {device}"
    )
    print(
        f"Data root       : {args.data_root}"
    )
    print(
        f"Task directory  : {task_dir}"
    )
    print(
        f"Dataset cache   : {cache_dir}"
    )
    print("=" * 80)

    for rep in range(
        args.repeats
    ):
        for fold in range(
            args.folds
        ):
            fold_seed = (
                args.seed
                + rep * 1000
                + fold
            )

            seed_everything(
                fold_seed
            )


            flat_fold = (
                rep * args.folds
                + fold
            )

            train_cache = (
                cache_dir
                / f"f{flat_fold}_train.data"
            )

            val_cache = (
                cache_dir
                / f"f{flat_fold}_val.data"
            )

            test_cache = (
                cache_dir
                / f"f{flat_fold}_test.data"
            )

            cache_paths = [
                train_cache,
                val_cache,
                test_cache,
            ]

            ckpt_path = (
                task_dir
                / f"HERBA_f{flat_fold}.pt"
            )

            if args.action == "test":
                missing = [
                    str(p)
                    for p in (
                        test_cache,
                        ckpt_path,
                    )
                    if not p.is_file()
                ]

                if missing:
                    raise FileNotFoundError(
                        "Released test files are missing:\n"
                        + "\n".join(missing)
                    )

                test_data = load_torch(
                    test_cache,
                    map_location="cpu",
                )

                train_data = None
                val_data = None

            else:
                use_existing_cache = (
                    not args.rebuild_cache
                    and all(
                        p.is_file()
                        for p in cache_paths
                    )
                )

                if use_existing_cache:
                    print(
                        f"[cache] using f{flat_fold} "
                        f"(rep={rep}, fold={fold})"
                    )

                    (
                        train_data,
                        val_data,
                        test_data,
                    ) = [
                        load_torch(
                            p,
                            map_location="cpu",
                        )
                        for p in cache_paths
                    ]

                else:
                    print(
                        f"[cache] building f{flat_fold} "
                        f"(rep={rep}, fold={fold})"
                    )

                    (
                        train_data,
                        val_data,
                        test_data,
                    ) = create_classification_dataset(
                        args.data_root,
                        "RNA",
                        rep,
                        fold,
                        args.folds,
                        args.val_fraction,
                        args.threshold,
                        args.rnafm_dim,
                    )

                    for obj, path in zip(
                        (
                            train_data,
                            val_data,
                            test_data,
                        ),
                        cache_paths,
                    ):
                        torch.save(
                            obj,
                            path,
                        )

            if args.action == "train":
                train_loader = (
                    torch.utils.data.DataLoader(
                        train_data,
                        batch_size=args.batch,
                        shuffle=True,
                        collate_fn=collate,
                    )
                )

                val_loader = (
                    torch.utils.data.DataLoader(
                        val_data,
                        batch_size=args.batch_test,
                        shuffle=False,
                        collate_fn=collate,
                    )
                )

                model = HERBA(
                    rnafm_dim=args.rnafm_dim
                ).to(device)

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=args.lr,
                )

                pos_weight = cls_pos_weight(
                    train_data,
                    device,
                )

                criterion = (
                    torch.nn.BCEWithLogitsLoss(
                        pos_weight=pos_weight
                    )
                )

                print(
                    "[class-weight] "
                    f"pos_weight="
                    f"{float(pos_weight.item()):.6f}"
                )

                best_bacc = -1.0
                best_loss = float("inf")
                best_epoch = -1
                stale = 0

                for epoch in range(
                    1,
                    args.epochs + 1,
                ):
                    train_cls_epoch(
                        model,
                        device,
                        train_loader,
                        optimizer,
                        criterion,
                        epoch,
                    )

                    vy, vp = predict_cls(
                        model,
                        device,
                        val_loader,
                    )

                    val_metrics = cls_metrics(
                        vy,
                        vp,
                        cutoff=args.prob_threshold,
                    )


                    vp_clip = np.clip(
                        vp,
                        1e-7,
                        1.0 - 1e-7,
                    )

                    val_loss = float(
                        -np.mean(
                            vy * np.log(vp_clip)
                            + (1.0 - vy)
                            * np.log(
                                1.0 - vp_clip
                            )
                        )
                    )

                    eps = 1e-12

                    improved = (
                        val_metrics["bacc"]
                        > best_bacc + eps
                        or (
                            abs(
                                val_metrics["bacc"]
                                - best_bacc
                            )
                            <= eps
                            and val_loss
                            < best_loss
                        )
                    )

                    if improved:
                        best_bacc = float(
                            val_metrics["bacc"]
                        )
                        best_loss = val_loss
                        best_epoch = epoch
                        stale = 0

                        atomic_save(
                            {
                                "epoch": epoch,
                                "best_val_bacc": (
                                    best_bacc
                                ),
                                "best_val_loss": (
                                    best_loss
                                ),
                                "model_state_dict": (
                                    model.state_dict()
                                ),
                                "seed": fold_seed,
                                "model": MODEL_NAME,
                                "task": task_name,
                                "split_version": (
                                    SPLIT_VERSION
                                ),
                                "threshold": (
                                    args.threshold
                                ),
                                "n_splits": (
                                    args.folds
                                ),
                                "repeat": rep,
                                "fold": fold,
                                "flat_fold": (
                                    flat_fold
                                ),
                                "pos_weight": float(
                                    pos_weight.item()
                                ),
                            },
                            ckpt_path,
                        )

                        print(
                            f"[VAL+] "
                            f"rep={rep} "
                            f"fold={fold} "
                            f"flat=f{flat_fold} "
                            f"epoch={epoch} "
                            f"BACC={best_bacc:.6f} "
                            f"loss={best_loss:.6f}"
                        )

                    else:
                        stale += 1

                    if (
                        stale
                        >= args.patience
                    ):
                        print(
                            f"[EARLY] "
                            f"rep={rep} "
                            f"fold={fold} "
                            f"flat=f{flat_fold} "
                            f"best_epoch="
                            f"{best_epoch}"
                        )
                        break

                del (
                    train_loader,
                    val_loader,
                    model,
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint not found: "
                    f"{ckpt_path}"
                )

            checkpoint = load_torch(
                ckpt_path,
                map_location=device,
            )

            model = HERBA(
                rnafm_dim=args.rnafm_dim
            ).to(device)

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ],
                strict=True,
            )

            test_loader = (
                torch.utils.data.DataLoader(
                    test_data,
                    batch_size=args.batch_test,
                    shuffle=False,
                    collate_fn=collate,
                )
            )

            y, prob = predict_cls(
                model,
                device,
                test_loader,
            )

            test_metrics = cls_metrics(
                y,
                prob,
                cutoff=args.prob_threshold,
            )

            row = {
                "dataset": "RNA",
                "rep": rep,
                "fold": fold,
                "flat_fold": (
                    flat_fold
                ),
                "model": MODEL_NAME,
                "best_epoch": (
                    checkpoint.get(
                        "epoch",
                        "",
                    )
                ),
                "val_bacc": (
                    checkpoint.get(
                        "best_val_bacc",
                        "",
                    )
                ),
                "val_loss": (
                    checkpoint.get(
                        "best_val_loss",
                        "",
                    )
                ),
                **{
                    k: float(v)
                    for k, v
                    in test_metrics.items()
                },
            }

            upsert_csv(
                result_path,
                row,
                [
                    "dataset",
                    "rep",
                    "fold",
                    "model",
                ],
                columns,
            )

            pred_path = (
                task_dir
                / f"preds_f{flat_fold}.csv"
            )

            with open(
                pred_path,
                "w",
                newline="",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "y_true",
                        "prob",
                        "pred",
                    ]
                )

                for yy, pp in zip(
                    y,
                    prob,
                ):
                    writer.writerow(
                        [
                            int(yy),
                            float(pp),
                            int(
                                pp
                                >= args.prob_threshold
                            ),
                        ]
                    )

            print(
                f"[TEST] f{flat_fold} "
                f"ACC={row['accuracy']:.6f} "
                f"BACC={row['bacc']:.6f} "
                f"AUC={row['auc']:.6f} "
                f"AUPRC={row['auprc']:.6f}"
            )

            del (
                model,
                test_loader,
                test_data,
            )

            if (
                train_data is not None
            ):
                del train_data

            if (
                val_data is not None
            ):
                del val_data

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nFinished.")
    print(
        f"Results saved to: "
        f"{result_path}"
    )


if __name__ == "__main__":
    main()
