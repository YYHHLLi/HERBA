#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HERBA regression cross-validation.

This script supports both 5-fold and 10-fold regression experiments.

Released directory layout
-------------------------
HERBA/
├── Checkpoint/
│   ├── regression_cv_5/
│   │   ├── HERBA_f0.pt
│   │   ├── HERBA_f1.pt
│   │   ├── ...
│   │   ├── dataset_cache/
│   │   │   ├── f0_train.data
│   │   │   ├── f0_val.data
│   │   │   ├── f0_test.data
│   │   │   └── ...
│   │   └── outer_test_results.csv
│   └── regression_cv_10/
│       └── ...
├── data/
├── models/
│   └── HERBA.py
├── create_data_fold.py
├── kmer.py
├── utils.py
└── regression_cv.py

Examples
--------
Train 5-fold regression:
    python regression_cv.py --action train --folds 5 --gpu 0

Test released 5-fold checkpoints:
    python regression_cv.py --action test --folds 5 --gpu 0

Train 10-fold regression:
    python regression_cv.py --action train --folds 10 --gpu 0

Test released 10-fold checkpoints:
    python regression_cv.py --action test --folds 10 --gpu 0
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
from sklearn.model_selection import KFold, train_test_split

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
from utils import (
    TestbedDataset,
    collate,
    predicting,
    rmse,
    mse,
    pearson,
    spearman,
    ci,
    rm2,
)


MODEL_NAME = "HERBA"
SPLIT_VERSION = "nested_v1"


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


def train_regression_epoch(
    model,
    device,
    loader,
    optimizer,
    epoch,
):
    model.train()
    loss_fn = torch.nn.MSELoss()

    running = 0.0
    n = 0

    for drug, target in loader:
        drug = drug.to(device)
        target = target.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model(
            drug,
            target,
        )

        y = drug.y.view(
            -1,
            1,
        ).float()

        loss = loss_fn(
            pred,
            y,
        )

        loss.backward()
        optimizer.step()

        running += float(
            loss.item()
        )
        n += 1

    mean_loss = running / max(
        1,
        n,
    )

    print(
        f"epoch={epoch} "
        f"train_mse={mean_loss:.6f}"
    )

    return mean_loss


def get_pair_folds(
    fpath,
    affinity,
    n_splits,
    rep,
):

    rows_all, cols_all = np.where(
        ~np.isnan(affinity)
    )

    fold_dir = os.path.join(
        fpath,
        "folds",
    )

    fold_file = os.path.join(
        fold_dir,
        (
            f"pair_kfold_"
            f"{n_splits}_rep{rep}.json"
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
            folds,
        )

    kf = KFold(
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
        in kf.split(idx)
    ]

    atomic_json_dump(
        folds,
        fold_file,
    )

    return (
        rows_all,
        cols_all,
        folds,
    )


def split_inner_val(
    outer_train_idx,
    rows_all,
    cols_all,
    affinity,
    rep,
    fold,
    val_fraction,
):


    outer_train_idx = np.asarray(
        outer_train_idx,
        dtype=int,
    )

    y = affinity[
        rows_all[outer_train_idx],
        cols_all[outer_train_idx],
    ]

    stratify = None

    try:
        n_bins = min(
            10,
            max(
                2,
                len(outer_train_idx) // 50,
            ),
        )

        bins = pd.qcut(
            y,
            q=n_bins,
            labels=False,
            duplicates="drop",
        )

        counts = pd.Series(
            bins
        ).value_counts()

        if (
            len(counts) >= 2
            and counts.min() >= 2
        ):
            stratify = np.asarray(
                bins
            )

    except Exception:
        stratify = None

    return train_test_split(
        outer_train_idx,
        test_size=val_fraction,
        random_state=(
            2026
            + rep * 1000
            + fold
        ),
        shuffle=True,
        stratify=stratify,
    )


def create_regression_dataset(
    data_root,
    dataset,
    rep,
    fold,
    n_splits,
    val_fraction,
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
        folds,
    ) = get_pair_folds(
        fpath,
        affinity,
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

    (
        train_idx,
        val_idx,
    ) = split_inner_val(
        outer_train_idx,
        rows_all,
        cols_all,
        affinity,
        rep,
        fold,
        val_fraction,
    )

    tr_r = rows_all[train_idx]
    tr_c = cols_all[train_idx]

    va_r = rows_all[val_idx]
    va_c = cols_all[val_idx]

    te_r = rows_all[test_idx]
    te_c = cols_all[test_idx]

    print(
        f"[split] rep={rep} "
        f"fold={fold}/{n_splits}: "
        f"train={len(train_idx)} "
        f"val={len(val_idx)} "
        f"test={len(test_idx)}"
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
            f"kmer_z_reg{n_splits}_"
            f"nested_r{rep}_f{fold}.npz"
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
            f"{dataset}_reg{n_splits}_"
            f"r{rep}_f{fold}_train"
        ),
        tr_r,
        tr_c,
        affinity[tr_r, tr_c],
        store,
        kmer_map,
        rnafm_dim,
    )

    val_data = make_dataset_from_pairs(
        (
            f"{dataset}_reg{n_splits}_"
            f"r{rep}_f{fold}_val"
        ),
        va_r,
        va_c,
        affinity[va_r, va_c],
        store,
        kmer_map,
        rnafm_dim,
    )

    test_data = make_dataset_from_pairs(
        (
            f"{dataset}_reg{n_splits}_"
            f"r{rep}_f{fold}_test"
        ),
        te_r,
        te_c,
        affinity[te_r, te_c],
        store,
        kmer_map,
        rnafm_dim,
    )

    return (
        train_data,
        val_data,
        test_data,
    )


def main():
    project_root = Path(
        __file__
    ).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "HERBA regression 5-fold/10-fold "
            "cross-validation."
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

    seed_everything(
        args.seed
    )

    device = torch.device(
        (
            f"cuda:{args.gpu}"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    task_name = (
        f"regression_cv_{args.folds}"
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
        "val_mse",
        "rmse",
        "mse",
        "pearson",
        "spearman",
        "ci",
        "rm2",
    ]

    print("=" * 80)
    print("HERBA regression CV")
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
                    ) = create_regression_dataset(
                        args.data_root,
                        "RNA",
                        rep,
                        fold,
                        args.folds,
                        args.val_fraction,
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

                best_val = float(
                    "inf"
                )
                best_epoch = -1
                stale = 0

                for epoch in range(
                    1,
                    args.epochs + 1,
                ):
                    train_regression_epoch(
                        model,
                        device,
                        train_loader,
                        optimizer,
                        epoch,
                    )

                    vy, vp = predicting(
                        model,
                        device,
                        val_loader,
                    )

                    vmse = float(
                        mse(vy, vp)
                    )

                    if vmse < best_val:
                        best_val = vmse
                        best_epoch = epoch
                        stale = 0

                        atomic_save(
                            {
                                "epoch": epoch,
                                "best_val_mse": (
                                    best_val
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
                                "n_splits": (
                                    args.folds
                                ),
                                "repeat": rep,
                                "fold": fold,
                                "flat_fold": (
                                    flat_fold
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
                            f"mse={vmse:.6f}"
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

            y, p = predicting(
                model,
                device,
                test_loader,
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
                "val_mse": (
                    checkpoint.get(
                        "best_val_mse",
                        "",
                    )
                ),
                "rmse": float(
                    rmse(y, p)
                ),
                "mse": float(
                    mse(y, p)
                ),
                "pearson": float(
                    pearson(y, p)
                ),
                "spearman": float(
                    spearman(y, p)
                ),
                "ci": float(
                    ci(y, p)
                ),
                "rm2": float(
                    rm2(y, p)
                ),
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
                        "y_pred",
                    ]
                )
                writer.writerows(
                    zip(y, p)
                )

            print(
                f"[TEST] f{flat_fold} "
                f"RMSE={row['rmse']:.6f} "
                f"PCC={row['pearson']:.6f} "
                f"SCC={row['spearman']:.6f}"
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
