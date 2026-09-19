#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HERBA strict cold-start regression.

Modes
-----
rna30:
    RNA sequence-cluster cold-start using MMseqs2:
    --min-seq-id 0.30 -c 0.80 --cov-mode 0

rna80:
    RNA sequence-cluster cold-start using MMseqs2:
    --min-seq-id 0.80 -c 0.80 --cov-mode 0

scaffold:
    Small-molecule Bemis-Murcko scaffold cold-start using RDKit.

Protocol
--------
1) Outer group-disjoint K-fold.
2) Inner validation is also group-disjoint from inner training.
3) Checkpoint / early stopping uses validation MSE only.
4) Outer test is evaluated only after the best validation checkpoint is loaded.
5) k-mer z-score statistics are fitted on INNER TRAIN RNA sequences only.
6) Default = 5 folds x 10 repeats.

Place directly under:
    /ifs/home/liyahan/projects/HERBA/F2_MSE_MSResidual/

Reuses:
    models/HERBA.py
    utils.py
    create_data_fold.py
    kmer.py
"""

import argparse
import csv
import json
import os
import pickle
import random
import shutil
import subprocess
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

from create_data_fold import (
    safe_mol_from_smiles,
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_torch(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


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
    with open(tmp, "w") as handle:
        json.dump(obj, handle, indent=2)
    os.replace(tmp, path)


def upsert_result(path, row):
    columns = [
        "dataset",
        "mode",
        "rep",
        "fold",
        "flat_fold",
        "model",
        "best_epoch",
        "val_mse",
        "n_train_pairs",
        "n_val_pairs",
        "n_test_pairs",
        "n_train_groups",
        "n_val_groups",
        "n_test_groups",
        "n_train_rna",
        "n_val_rna",
        "n_test_rna",
        "n_train_drug",
        "n_val_drug",
        "n_test_drug",
        "rmse",
        "mse",
        "pearson",
        "spearman",
        "ci",
        "rm2",
    ]

    path = str(path)
    previous = []

    if os.path.isfile(path):
        with open(path, "r", newline="") as handle:
            previous = list(csv.DictReader(handle))

    key_cols = [
        "dataset",
        "mode",
        "rep",
        "fold",
        "model",
    ]
    key = tuple(str(row[k]) for k in key_cols)

    kept = []
    for old in previous:
        old_key = tuple(str(old.get(k, "")) for k in key_cols)
        if old_key != key:
            kept.append(old)

    kept.append({
        column: row.get(column, "")
        for column in columns
    })

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(kept)


def load_raw(data_root, dataset="RNA"):
    fpath = os.path.join(data_root, dataset)

    with open(os.path.join(fpath, "ligand.json"), "r") as handle:
        ligands = json.load(handle, object_pairs_hook=OrderedDict)

    with open(os.path.join(fpath, "rna.json"), "r") as handle:
        proteins = json.load(handle, object_pairs_hook=OrderedDict)

    with open(os.path.join(fpath, "Y"), "rb") as handle:
        affinity = pickle.load(handle, encoding="latin1")

    affinity = np.asarray(affinity)
    if affinity.shape != (len(ligands), len(proteins)):
        raise RuntimeError(
            f"Y shape={affinity.shape}, ligands={len(ligands)}, RNAs={len(proteins)}"
        )

    return fpath, ligands, proteins, affinity


def build_feature_store(data_root, dataset, ligands, proteins, rnafm_dim):
    fpath = os.path.join(data_root, dataset)
    aln_path = os.path.join(fpath, "aln")
    pconsc_path = os.path.join(fpath, "pconsc4")
    rnafm_path = os.path.join(fpath, "rnafm")

    if not os.path.isdir(aln_path):
        raise RuntimeError(f"Missing aln directory: {aln_path}")
    if not os.path.isdir(pconsc_path):
        raise RuntimeError(f"Missing pconsc4 directory: {pconsc_path}")

    ligand_keys = list(ligands.keys())
    ligand_values = list(ligands.values())
    protein_keys = list(proteins.keys())

    drug_df = pd.read_csv(os.path.join(data_root, "drug_embeddings.csv"))
    drug_df["ID"] = drug_df["ID"].astype(str)
    drug_emb_map = {
        row["ID"]: row.iloc[2:].values.astype(np.float32)
        for _, row in drug_df.iterrows()
    }

    print("[features] building molecule graphs...")
    smile_graph = {s: smile_to_graph(s) for s in ligand_values}
    smile_tensor = {s: label_smiles(s, 100) for s in ligand_values}

    seq_for_key = {
        key: clean_rna_seq(value)
        for key, value in proteins.items()
    }

    print("[features] building RNA graphs/features...")
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


def _mmseqs_identity_for_mode(mode):
    if mode == "rna30":
        return 0.30
    if mode == "rna80":
        return 0.80
    raise ValueError(mode)


def _sanitize_seq_for_mmseqs(seq):

    seq = clean_rna_seq(seq).replace("U", "T")
    seq = "".join(ch for ch in seq if ch in "ACGTN")
    return seq if seq else "N"


def get_or_make_rna_clusters(
    mode,
    proteins,
    cache_dir,
    mmseqs_bin="mmseqs",
    coverage=0.80,
):
    identity = _mmseqs_identity_for_mode(mode)
    os.makedirs(cache_dir, exist_ok=True)

    mapping_csv = os.path.join(
        cache_dir,
        f"{mode}_mmseqs_id{identity:.2f}_cov{coverage:.2f}_mapping.csv",
    )

    if os.path.isfile(mapping_csv):
        df = pd.read_csv(mapping_csv, dtype={"rna_id": str, "cluster": str})
        mapping = dict(zip(df["rna_id"], df["cluster"]))
        missing = [str(k) for k in proteins.keys() if str(k) not in mapping]
        if missing:
            raise RuntimeError(
                f"Cached MMseqs mapping incomplete: {len(missing)} RNAs missing."
            )
        print(
            f"[groups] loaded {mode}: "
            f"{df['cluster'].nunique()} RNA clusters"
        )
        return mapping

    executable = shutil.which(mmseqs_bin)
    if executable is None:
        raise RuntimeError(
            f"MMseqs2 executable '{mmseqs_bin}' not found.\n"
            "Check with: mmseqs version\n"
            "or pass --mmseqs-bin /full/path/to/mmseqs"
        )

    work_dir = os.path.join(cache_dir, f"{mode}_mmseqs_work")
    os.makedirs(work_dir, exist_ok=True)

    fasta_path = os.path.join(work_dir, "rna_unique.fasta")
    prefix = os.path.join(work_dir, "cluster")
    tmp_dir = os.path.join(work_dir, "tmp")
    cluster_tsv = prefix + "_cluster.tsv"

    synthetic_to_rna = {}
    with open(fasta_path, "w") as handle:
        for i, (rna_id, seq) in enumerate(proteins.items()):
            sid = f"RNAIDX_{i:06d}"
            synthetic_to_rna[sid] = str(rna_id)
            handle.write(f">{sid}\n{_sanitize_seq_for_mmseqs(seq)}\n")

    for suffix in ("_cluster.tsv", "_rep_seq.fasta", "_all_seqs.fasta"):
        p = prefix + suffix
        if os.path.isfile(p):
            os.remove(p)

    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        executable,
        "easy-cluster",
        fasta_path,
        prefix,
        tmp_dir,
        "--min-seq-id", str(identity),
        "-c", str(coverage),
        "--cov-mode", "0",
    ]
    print("[MMseqs2]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not os.path.isfile(cluster_tsv):
        raise RuntimeError(f"MMseqs2 cluster TSV not found: {cluster_tsv}")

    representative_of = {}
    with open(cluster_tsv, "r") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            rep_sid, member_sid = fields[0], fields[1]
            representative_of[member_sid] = rep_sid

    rows = []
    for sid, rna_id in synthetic_to_rna.items():
        rep_sid = representative_of.get(sid, sid)
        rows.append({
            "rna_id": str(rna_id),
            "cluster": f"{mode}::{rep_sid}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(mapping_csv, index=False)

    print(
        f"[groups] generated {mode}: "
        f"{len(df)} RNAs -> {df['cluster'].nunique()} clusters"
    )
    print(f"[groups] saved: {mapping_csv}")

    return dict(zip(df["rna_id"], df["cluster"]))


def scaffold_group(smiles, ligand_id):
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return f"INVALID::{ligand_id}"

    canonical = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )
    except Exception:
        scaffold = ""

    if scaffold:
        scaf_mol = Chem.MolFromSmiles(scaffold)
        if scaf_mol is not None:
            scaffold = Chem.MolToSmiles(
                scaf_mol,
                canonical=True,
                isomericSmiles=False,
            )
        return f"MURCKO::{scaffold}"


    return f"ACYCLIC::{canonical}"


def get_or_make_scaffold_groups(ligands, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    mapping_csv = os.path.join(
        cache_dir,
        "bemis_murcko_scaffold_mapping.csv",
    )

    if os.path.isfile(mapping_csv):
        df = pd.read_csv(
            mapping_csv,
            dtype={"ligand_id": str, "scaffold_group": str},
        )
        mapping = dict(zip(df["ligand_id"], df["scaffold_group"]))
        missing = [str(k) for k in ligands.keys() if str(k) not in mapping]
        if missing:
            raise RuntimeError(
                f"Cached scaffold mapping incomplete: {len(missing)} ligands missing."
            )
        print(
            f"[groups] loaded scaffold mapping: "
            f"{df['scaffold_group'].nunique()} groups"
        )
        return mapping

    rows = []
    for ligand_id, smiles in ligands.items():
        rows.append({
            "ligand_id": str(ligand_id),
            "smiles": smiles,
            "scaffold_group": scaffold_group(smiles, str(ligand_id)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(mapping_csv, index=False)

    print(
        f"[groups] generated scaffolds: "
        f"{len(df)} molecules -> "
        f"{df['scaffold_group'].nunique()} groups"
    )
    print(f"[groups] saved: {mapping_csv}")

    return dict(zip(df["ligand_id"], df["scaffold_group"]))


def randomized_balanced_group_folds(pair_groups, n_splits, seed):
    pair_groups = np.asarray(pair_groups, dtype=object)
    unique_groups, inverse = np.unique(pair_groups, return_inverse=True)

    if len(unique_groups) < n_splits:
        raise RuntimeError(
            f"Only {len(unique_groups)} groups for {n_splits} folds."
        )

    counts = np.bincount(inverse)
    rng = np.random.RandomState(seed)
    jitter = rng.random(len(unique_groups))

    order = sorted(
        range(len(unique_groups)),
        key=lambda i: (-int(counts[i]), float(jitter[i])),
    )

    fold_load = np.zeros(n_splits, dtype=np.int64)
    fold_groups = [[] for _ in range(n_splits)]

    for i in order:
        candidates = np.flatnonzero(fold_load == fold_load.min())
        chosen = int(rng.choice(candidates))
        fold_groups[chosen].append(unique_groups[i])
        fold_load[chosen] += int(counts[i])

    folds = []
    for groups_here in fold_groups:
        mask = np.isin(pair_groups, np.asarray(groups_here, dtype=object))
        folds.append(np.flatnonzero(mask).astype(int))

    merged = np.concatenate(folds)
    if len(merged) != len(pair_groups):
        raise RuntimeError("Outer folds do not cover every pair.")
    if len(np.unique(merged)) != len(pair_groups):
        raise RuntimeError("Some pair appears in more than one outer fold.")

    return folds


def get_or_make_outer_folds(
    pair_groups,
    mode,
    rep,
    n_splits,
    split_dir,
    seed,
):
    os.makedirs(split_dir, exist_ok=True)
    path = os.path.join(
        split_dir,
        f"{mode}_{n_splits}fold_rep{rep}_outer.json",
    )

    if os.path.isfile(path):
        with open(path, "r") as handle:
            obj = json.load(handle)

        if int(obj["n_pairs"]) != len(pair_groups):
            raise RuntimeError(f"Stale split file: {path}")

        folds = [np.asarray(x, dtype=int) for x in obj["folds"]]
        if len(folds) != n_splits:
            raise RuntimeError(f"Wrong fold count in: {path}")
        return folds

    folds = randomized_balanced_group_folds(
        pair_groups,
        n_splits=n_splits,
        seed=seed,
    )

    atomic_json_dump(
        {
            "mode": mode,
            "rep": rep,
            "n_splits": n_splits,
            "seed": seed,
            "n_pairs": len(pair_groups),
            "n_groups": len(set(pair_groups.tolist())),
            "folds": [x.tolist() for x in folds],
        },
        path,
    )
    return folds


def group_disjoint_inner_split(
    outer_train_idx,
    pair_groups,
    val_fraction,
    seed,
):
    outer_train_idx = np.asarray(outer_train_idx, dtype=int)
    groups = np.asarray(pair_groups, dtype=object)[outer_train_idx]

    if len(np.unique(groups)) < 2:
        raise RuntimeError(
            "Outer train has fewer than two groups; cannot create validation."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=seed,
    )

    rel_train, rel_val = next(
        splitter.split(
            np.zeros(len(outer_train_idx)),
            groups=groups,
        )
    )

    train_idx = outer_train_idx[rel_train]
    val_idx = outer_train_idx[rel_val]

    train_groups = set(np.asarray(pair_groups, dtype=object)[train_idx].tolist())
    val_groups = set(np.asarray(pair_groups, dtype=object)[val_idx].tolist())

    if not train_groups.isdisjoint(val_groups):
        raise RuntimeError("BUG: inner train/val group overlap.")

    return train_idx, val_idx


def make_dataset_from_pairs(
    dataset_name,
    pair_idx,
    rows_all,
    cols_all,
    affinity,
    store,
    kmer_map,
):
    pair_idx = np.asarray(pair_idx, dtype=int)
    rows = rows_all[pair_idx]
    cols = cols_all[pair_idx]

    ligand_keys = store["ligand_keys"]
    ligand_values = store["ligand_values"]
    protein_keys = store["protein_keys"]

    keys = [protein_keys[int(col)] for col in cols]

    return TestbedDataset(
        root="/tmp",
        dataset=dataset_name,
        xd=[ligand_values[int(row)] for row in rows],
        xt=[
            seq_cat(store["seq_for_key"][protein_keys[int(col)]])
            for col in cols
        ],
        y=affinity[rows, cols],
        smile_graph=store["smile_graph"],
        smile_tensor=store["smile_tensor"],
        target_graph={
            key: store["target_graph"][key][:3]
            for key in keys
        },
        target_key=keys,
        kmer_map=kmer_map,
        rnafm_map={
            key: store["target_graph"][key][3]
            for key in keys
        },
        ligand_ids=np.asarray(ligand_keys)[rows],
        drug_emb_map=store["drug_emb_map"],
    )


def make_kmer_map_for_split(
    train_idx,
    val_idx,
    test_idx,
    cols_all,
    store,
    z_path,
):
    protein_keys = store["protein_keys"]
    train_cols = cols_all[np.asarray(train_idx, dtype=int)]

    unique_train_keys = sorted(
        {protein_keys[int(col)] for col in train_cols}
    )
    if not unique_train_keys:
        raise RuntimeError("No inner-train RNA for k-mer fitting.")

    if not os.path.isfile(z_path):
        train_seqs = [
            store["seq_for_key"][key]
            for key in unique_train_keys
        ]
        fit_kmer_zscore(train_seqs, save_path=z_path)

    mean, std = load_kmer_zscore(z_path)

    all_idx = np.concatenate([
        np.asarray(train_idx, dtype=int),
        np.asarray(val_idx, dtype=int),
        np.asarray(test_idx, dtype=int),
    ])

    needed_keys = sorted(
        {
            protein_keys[int(col)]
            for col in cols_all[all_idx]
        }
    )

    return {
        str(key): zscore(
            seq_to_kmer_freq(store["seq_for_key"][key]),
            mean,
            std,
        )
        for key in needed_keys
    }


def split_metadata(
    train_idx,
    val_idx,
    test_idx,
    pair_groups,
    rows_all,
    cols_all,
):
    pair_groups = np.asarray(pair_groups, dtype=object)

    tr_g = set(pair_groups[train_idx].tolist())
    va_g = set(pair_groups[val_idx].tolist())
    te_g = set(pair_groups[test_idx].tolist())

    if not tr_g.isdisjoint(va_g):
        raise RuntimeError("Train/val strict-group leakage.")
    if not tr_g.isdisjoint(te_g):
        raise RuntimeError("Train/test strict-group leakage.")
    if not va_g.isdisjoint(te_g):
        raise RuntimeError("Val/test strict-group leakage.")

    return {
        "n_train_pairs": int(len(train_idx)),
        "n_val_pairs": int(len(val_idx)),
        "n_test_pairs": int(len(test_idx)),
        "n_train_groups": int(len(tr_g)),
        "n_val_groups": int(len(va_g)),
        "n_test_groups": int(len(te_g)),
        "n_train_rna": int(len(set(cols_all[train_idx].tolist()))),
        "n_val_rna": int(len(set(cols_all[val_idx].tolist()))),
        "n_test_rna": int(len(set(cols_all[test_idx].tolist()))),
        "n_train_drug": int(len(set(rows_all[train_idx].tolist()))),
        "n_val_drug": int(len(set(rows_all[val_idx].tolist()))),
        "n_test_drug": int(len(set(rows_all[test_idx].tolist()))),
    }


def train_one_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_fn = torch.nn.MSELoss()

    running = 0.0
    n_batch = 0

    for drug, target in train_loader:
        drug = drug.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(drug, target)
        labels = drug.y.view(-1, 1).float()

        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()

        running += float(loss.item())
        n_batch += 1

    mean_loss = running / max(1, n_batch)
    print(f"epoch={epoch} train_mse={mean_loss:.6f}")
    return mean_loss


def print_repeat_summary(result_path, n_splits):
    if not os.path.isfile(result_path):
        return

    df = pd.read_csv(result_path)
    if df.empty:
        return

    metrics = ["rmse","mse","pearson","spearman","ci","rm2"]
    complete_reps = []

    for rep, group in df.groupby("rep"):
        if set(range(n_splits)).issubset(
            set(group["fold"].astype(int).tolist())
        ):
            complete_reps.append(int(rep))

    if not complete_reps:
        return

    per_rep = (
        df[df["rep"].isin(complete_reps)]
        .groupby("rep")[metrics]
        .mean()
    )

    print("\n================ SUMMARY ================")
    print("complete repeats:", complete_reps)
    for col in metrics:
        print(
            f"{col:9s} "
            f"{per_rep[col].mean():.4f} ± "
            f"{per_rep[col].std(ddof=1):.4f}"
        )
    print("=========================================\n")


def main():
    from pathlib import Path

    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "HERBA strict cold-start regression. "
            "RNA modes use MMseqs2 sequence clusters; "
            "scaffold mode uses Bemis-Murcko scaffold groups."
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "rna30",
            "rna80",
            "scaffold",
        ],
    )

    parser.add_argument(
        "--action",
        choices=[
            "prepare",
            "train",
            "test",
        ],
        default="train",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help=(
            "Number of repeated CV runs for training/preparation. "
            "The strict benchmark uses 10 by default."
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
        "--min-delta",
        type=float,
        default=0.0,
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
        default=os.environ.get(
            "HERBA_DATA_DIR",
            str(project_root / "data"),
        ),
        help="HERBA data directory. Default: ./data",
    )

    parser.add_argument(
        "--checkpoint-root",
        default=str(
            project_root / "Checkpoint"
        ),
        help=(
            "Root directory containing released HERBA "
            "checkpoint bundles."
        ),
    )

    parser.add_argument(
        "--mmseqs-bin",
        default="mmseqs",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help=(
            "Rebuild train/val/test .data files during training "
            "even if public-format cache files already exist."
        ),
    )

    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError(
            "--folds must be >= 2"
        )
    if args.repeats < 1:
        raise ValueError(
            "--repeats must be >= 1"
        )
    if not (
        0.0
        < args.val_fraction
        < 1.0
    ):
        raise ValueError(
            "--val-fraction must be in (0,1)"
        )

    task_name_map = {
        "rna30": "rna_similarity_30",
        "rna80": "rna_similarity_80",
        "scaffold": "molecular_scaffold",
    }
    task_name = task_name_map[
        args.mode
    ]

    task_dir = (
        Path(args.checkpoint_root)
        / task_name
    )
    cache_dir = (
        task_dir
        / "dataset_cache"
    )


    group_cache_dir = (
        task_dir
        / "_group_cache"
    )
    split_dir = (
        task_dir
        / "_split_cache"
    )
    preprocess_dir = (
        task_dir
        / "_preprocess"
    )
    manifest_dir = (
        task_dir
        / "_split_manifests"
    )

    task_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        (
            f"cuda:{args.gpu}"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    result_path = (
        task_dir
        / "outer_test_results.csv"
    )

    print("=" * 80)
    print("HERBA strict cold-start regression")
    print(
        f"Action          : {args.action}"
    )
    print(
        f"Mode            : {args.mode}"
    )
    print(
        f"Task directory  : {task_dir}"
    )
    print(
        f"Dataset cache   : {cache_dir}"
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
    print("=" * 80)

    if args.action == "test":
        checkpoint_paths = sorted(
            task_dir.glob(
                "HERBA_f*.pt"
            ),
            key=lambda p: int(
                p.stem.split("_f")[-1]
            ),
        )

        if not checkpoint_paths:
            raise FileNotFoundError(
                f"No HERBA_f*.pt files found in: "
                f"{task_dir}"
            )

        for checkpoint_path in checkpoint_paths:
            flat_fold = int(
                checkpoint_path.stem.split(
                    "_f"
                )[-1]
            )

            rep = (
                flat_fold
                // args.folds
            )
            fold = (
                flat_fold
                % args.folds
            )

            test_cache = (
                cache_dir
                / f"f{flat_fold}_test.data"
            )
            meta_path = (
                cache_dir
                / f"f{flat_fold}_meta.json"
            )

            if not test_cache.is_file():
                raise FileNotFoundError(
                    "Checkpoint/cache mismatch:\n"
                    f"  checkpoint: {checkpoint_path}\n"
                    f"  missing   : {test_cache}"
                )

            test_data = load_torch(
                test_cache,
                map_location="cpu",
            )

            checkpoint = load_torch(
                checkpoint_path,
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

            y_true, y_pred = predicting(
                model,
                device,
                test_loader,
            )

            meta = {
                "n_train_pairs": "",
                "n_val_pairs": "",
                "n_test_pairs": "",
                "n_train_groups": "",
                "n_val_groups": "",
                "n_test_groups": "",
                "n_train_rna": "",
                "n_val_rna": "",
                "n_test_rna": "",
                "n_train_drug": "",
                "n_val_drug": "",
                "n_test_drug": "",
            }

            if meta_path.is_file():
                with open(
                    meta_path,
                    "r",
                ) as handle:
                    meta.update(
                        json.load(handle)
                    )

            row = {
                "dataset": "RNA",
                "mode": args.mode,
                "rep": rep,
                "fold": fold,
                "flat_fold": flat_fold,
                "model": MODEL_NAME,
                "best_epoch": checkpoint.get(
                    "epoch",
                    "",
                ),
                "val_mse": checkpoint.get(
                    "best_val_mse",
                    "",
                ),
                **meta,
                "rmse": float(
                    rmse(
                        y_true,
                        y_pred,
                    )
                ),
                "mse": float(
                    mse(
                        y_true,
                        y_pred,
                    )
                ),
                "pearson": float(
                    pearson(
                        y_true,
                        y_pred,
                    )
                ),
                "spearman": float(
                    spearman(
                        y_true,
                        y_pred,
                    )
                ),
                "ci": float(
                    ci(
                        y_true,
                        y_pred,
                    )
                ),
                "rm2": float(
                    rm2(
                        y_true,
                        y_pred,
                    )
                ),
            }

            upsert_result(
                result_path,
                row,
            )

            pred_path = (
                task_dir
                / f"preds_f{flat_fold}.csv"
            )

            with open(
                pred_path,
                "w",
                newline="",
            ) as handle:
                writer = csv.writer(
                    handle
                )
                writer.writerow(
                    [
                        "y_true",
                        "y_pred",
                    ]
                )
                writer.writerows(
                    zip(
                        y_true,
                        y_pred,
                    )
                )

            print(
                f"[TEST] {task_name} "
                f"f{flat_fold} "
                f"RMSE={row['rmse']:.6f} "
                f"PCC={row['pearson']:.6f} "
                f"SCC={row['spearman']:.6f}"
            )

            del (
                model,
                test_loader,
                test_data,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print_repeat_summary(
            str(result_path),
            n_splits=args.folds,
        )

        print(
            f"Results saved to: "
            f"{result_path}"
        )
        return


    for directory in (
        group_cache_dir,
        split_dir,
        preprocess_dir,
        manifest_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    (
        _,
        ligands,
        proteins,
        affinity,
    ) = load_raw(
        args.data_root,
        "RNA",
    )

    rows_all, cols_all = np.where(
        ~np.isnan(affinity)
    )

    ligand_keys = list(
        ligands.keys()
    )
    protein_keys = list(
        proteins.keys()
    )

    if args.mode in (
        "rna30",
        "rna80",
    ):
        rna_cluster_map = (
            get_or_make_rna_clusters(
                args.mode,
                proteins,
                str(group_cache_dir),
                mmseqs_bin=(
                    args.mmseqs_bin
                ),
                coverage=args.coverage,
            )
        )

        entity_group = {
            i: rna_cluster_map[
                str(key)
            ]
            for i, key
            in enumerate(
                protein_keys
            )
        }

        pair_groups = np.asarray(
            [
                entity_group[
                    int(col)
                ]
                for col in cols_all
            ],
            dtype=object,
        )

        print(
            f"[strict] {args.mode}: "
            f"{len(protein_keys)} RNAs -> "
            f"{len(set(entity_group.values()))} "
            f"sequence clusters; "
            f"{len(rows_all)} observed pairs"
        )

    else:
        scaffold_map = (
            get_or_make_scaffold_groups(
                ligands,
                str(group_cache_dir),
            )
        )

        entity_group = {
            i: scaffold_map[
                str(key)
            ]
            for i, key
            in enumerate(
                ligand_keys
            )
        }

        pair_groups = np.asarray(
            [
                entity_group[
                    int(row)
                ]
                for row in rows_all
            ],
            dtype=object,
        )

        print(
            f"[strict] scaffold: "
            f"{len(ligand_keys)} molecules -> "
            f"{len(set(entity_group.values()))} "
            f"scaffold groups; "
            f"{len(rows_all)} observed pairs"
        )

    if args.action == "prepare":
        for rep in range(
            args.repeats
        ):
            folds = (
                get_or_make_outer_folds(
                    pair_groups,
                    mode=args.mode,
                    rep=rep,
                    n_splits=args.folds,
                    split_dir=str(
                        split_dir
                    ),
                    seed=(
                        args.seed
                        + rep * 1000
                    ),
                )
            )

            print(
                f"[prepare] rep={rep}: "
                f"pair_sizes="
                f"{[len(x) for x in folds]}, "
                f"group_sizes="
                f"{[len(set(pair_groups[x].tolist())) for x in folds]}"
            )

        print(
            "[prepare] done."
        )
        return

    store = build_feature_store(
        args.data_root,
        "RNA",
        ligands,
        proteins,
        args.rnafm_dim,
    )

    for rep in range(
        args.repeats
    ):
        outer_folds = (
            get_or_make_outer_folds(
                pair_groups,
                mode=args.mode,
                rep=rep,
                n_splits=args.folds,
                split_dir=str(
                    split_dir
                ),
                seed=(
                    args.seed
                    + rep * 1000
                ),
            )
        )

        for fold in range(
            args.folds
        ):
            run_seed = (
                args.seed
                + rep * 1000
                + fold
            )

            seed_everything(
                run_seed
            )

            start_time = (
                time.time()
            )

            flat_fold = (
                rep * args.folds
                + fold
            )

            test_idx = np.asarray(
                outer_folds[fold],
                dtype=int,
            )

            outer_train_idx = (
                np.concatenate(
                    [
                        np.asarray(
                            one_fold,
                            dtype=int,
                        )
                        for i, one_fold
                        in enumerate(
                            outer_folds
                        )
                        if i != fold
                    ]
                )
            )

            (
                train_idx,
                val_idx,
            ) = (
                group_disjoint_inner_split(
                    outer_train_idx,
                    pair_groups,
                    val_fraction=(
                        args.val_fraction
                    ),
                    seed=run_seed,
                )
            )

            meta = split_metadata(
                train_idx,
                val_idx,
                test_idx,
                pair_groups,
                rows_all,
                cols_all,
            )

            print(
                f"\n[strict-{args.mode}] "
                f"rep={rep} "
                f"fold={fold}/{args.folds} "
                f"flat=f{flat_fold}: "
                f"pairs="
                f"{meta['n_train_pairs']}/"
                f"{meta['n_val_pairs']}/"
                f"{meta['n_test_pairs']}; "
                f"groups="
                f"{meta['n_train_groups']}/"
                f"{meta['n_val_groups']}/"
                f"{meta['n_test_groups']}"
            )

            manifest_path = (
                manifest_dir
                / f"split_manifest_f{flat_fold}.csv"
            )

            if not manifest_path.is_file():
                manifest_rows = []

                for split_name, idx_arr in (
                    (
                        "train",
                        train_idx,
                    ),
                    (
                        "val",
                        val_idx,
                    ),
                    (
                        "test",
                        test_idx,
                    ),
                ):
                    for pidx in idx_arr:
                        manifest_rows.append({
                            "split": split_name,
                            "pair_idx": int(
                                pidx
                            ),
                            "drug_row": int(
                                rows_all[pidx]
                            ),
                            "rna_col": int(
                                cols_all[pidx]
                            ),
                            "drug_id": str(
                                ligand_keys[
                                    int(
                                        rows_all[
                                            pidx
                                        ]
                                    )
                                ]
                            ),
                            "rna_id": str(
                                protein_keys[
                                    int(
                                        cols_all[
                                            pidx
                                        ]
                                    )
                                ]
                            ),
                            "strict_group": str(
                                pair_groups[
                                    pidx
                                ]
                            ),
                        })

                pd.DataFrame(
                    manifest_rows
                ).to_csv(
                    manifest_path,
                    index=False,
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
            meta_path = (
                cache_dir
                / f"f{flat_fold}_meta.json"
            )

            cache_paths = [
                train_cache,
                val_cache,
                test_cache,
            ]

            use_existing_cache = (
                not args.rebuild_cache
                and all(
                    path.is_file()
                    for path
                    in cache_paths
                )
            )

            if use_existing_cache:
                print(
                    f"[cache] using f{flat_fold}"
                )

                (
                    train_data,
                    val_data,
                    test_data,
                ) = [
                    load_torch(
                        path,
                        map_location="cpu",
                    )
                    for path
                    in cache_paths
                ]

            else:
                print(
                    f"[cache] building f{flat_fold}"
                )

                z_path = (
                    preprocess_dir
                    / (
                        f"kmer_z_"
                        f"{args.mode}_"
                        f"{args.folds}fold_"
                        f"r{rep}_f{fold}.npz"
                    )
                )

                kmer_map = (
                    make_kmer_map_for_split(
                        train_idx,
                        val_idx,
                        test_idx,
                        cols_all,
                        store,
                        str(z_path),
                    )
                )

                train_data = (
                    make_dataset_from_pairs(
                        (
                            f"RNA_"
                            f"{args.mode}_"
                            f"r{rep}_f{fold}_"
                            f"train"
                        ),
                        train_idx,
                        rows_all,
                        cols_all,
                        affinity,
                        store,
                        kmer_map,
                    )
                )

                val_data = (
                    make_dataset_from_pairs(
                        (
                            f"RNA_"
                            f"{args.mode}_"
                            f"r{rep}_f{fold}_"
                            f"val"
                        ),
                        val_idx,
                        rows_all,
                        cols_all,
                        affinity,
                        store,
                        kmer_map,
                    )
                )

                test_data = (
                    make_dataset_from_pairs(
                        (
                            f"RNA_"
                            f"{args.mode}_"
                            f"r{rep}_f{fold}_"
                            f"test"
                        ),
                        test_idx,
                        rows_all,
                        cols_all,
                        affinity,
                        store,
                        kmer_map,
                    )
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

            atomic_json_dump(
                meta,
                meta_path,
            )

            checkpoint_path = (
                task_dir
                / f"HERBA_f{flat_fold}.pt"
            )

            train_loader = (
                torch.utils.data.DataLoader(
                    train_data,
                    batch_size=args.batch,
                    shuffle=True,
                    collate_fn=collate,

                    drop_last=(
                        len(train_data)
                        % args.batch
                        == 1
                    ),
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

            best_val_mse = float(
                "inf"
            )
            best_epoch = -1
            stale_epochs = 0

            for epoch in range(
                1,
                args.epochs + 1,
            ):
                train_one_epoch(
                    model,
                    device,
                    train_loader,
                    optimizer,
                    epoch,
                )

                val_y, val_pred = (
                    predicting(
                        model,
                        device,
                        val_loader,
                    )
                )

                val_mse = float(
                    mse(
                        val_y,
                        val_pred,
                    )
                )

                if (
                    val_mse
                    < best_val_mse
                    - args.min_delta
                ):
                    best_val_mse = (
                        val_mse
                    )
                    best_epoch = epoch
                    stale_epochs = 0

                    atomic_save(
                        {
                            "epoch": epoch,
                            "best_val_mse": (
                                best_val_mse
                            ),
                            "model_state_dict": (
                                model.state_dict()
                            ),
                            "seed": run_seed,
                            "model": MODEL_NAME,
                            "task": task_name,
                            "mode": args.mode,
                            "n_splits": (
                                args.folds
                            ),
                            "repeat": rep,
                            "fold": fold,
                            "flat_fold": (
                                flat_fold
                            ),
                            "rna_identity": (
                                _mmseqs_identity_for_mode(
                                    args.mode
                                )
                                if args.mode
                                in (
                                    "rna30",
                                    "rna80",
                                )
                                else None
                            ),
                            "coverage": (
                                args.coverage
                                if args.mode
                                in (
                                    "rna30",
                                    "rna80",
                                )
                                else None
                            ),
                            "scaffold": (
                                "Bemis-Murcko"
                                if args.mode
                                == "scaffold"
                                else None
                            ),
                        },
                        checkpoint_path,
                    )

                    print(
                        f"[VAL+] "
                        f"mode={args.mode} "
                        f"rep={rep} "
                        f"fold={fold} "
                        f"flat=f{flat_fold} "
                        f"epoch={epoch} "
                        f"mse={val_mse:.6f}"
                    )

                else:
                    stale_epochs += 1

                if (
                    stale_epochs
                    >= args.patience
                ):
                    print(
                        f"[EARLY] "
                        f"mode={args.mode} "
                        f"rep={rep} "
                        f"fold={fold}; "
                        f"best_epoch="
                        f"{best_epoch}; "
                        f"best_val_mse="
                        f"{best_val_mse:.6f}"
                    )
                    break

            del (
                model,
                train_loader,
                val_loader,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint not found: "
                    f"{checkpoint_path}"
                )

            checkpoint = load_torch(
                checkpoint_path,
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

            y_true, y_pred = predicting(
                model,
                device,
                test_loader,
            )

            row = {
                "dataset": "RNA",
                "mode": args.mode,
                "rep": rep,
                "fold": fold,
                "flat_fold": (
                    flat_fold
                ),
                "model": MODEL_NAME,
                "best_epoch": int(
                    checkpoint[
                        "epoch"
                    ]
                ),
                "val_mse": float(
                    checkpoint[
                        "best_val_mse"
                    ]
                ),
                **meta,
                "rmse": float(
                    rmse(
                        y_true,
                        y_pred,
                    )
                ),
                "mse": float(
                    mse(
                        y_true,
                        y_pred,
                    )
                ),
                "pearson": float(
                    pearson(
                        y_true,
                        y_pred,
                    )
                ),
                "spearman": float(
                    spearman(
                        y_true,
                        y_pred,
                    )
                ),
                "ci": float(
                    ci(
                        y_true,
                        y_pred,
                    )
                ),
                "rm2": float(
                    rm2(
                        y_true,
                        y_pred,
                    )
                ),
            }

            upsert_result(
                result_path,
                row,
            )

            pred_path = (
                task_dir
                / f"preds_f{flat_fold}.csv"
            )

            with open(
                pred_path,
                "w",
                newline="",
            ) as handle:
                writer = csv.writer(
                    handle
                )
                writer.writerow(
                    [
                        "y_true",
                        "y_pred",
                    ]
                )
                writer.writerows(
                    zip(
                        y_true,
                        y_pred,
                    )
                )

            print(
                f"[TEST] {task_name} "
                f"f{flat_fold} "
                f"RMSE={row['rmse']:.6f} "
                f"PCC={row['pearson']:.6f} "
                f"SCC={row['spearman']:.6f}"
            )

            print(
                f"fold elapsed: "
                f"{time.time() - start_time:.1f}s"
            )

            del (
                model,
                test_loader,
                train_data,
                val_data,
                test_data,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print_repeat_summary(
        str(result_path),
        n_splits=args.folds,
    )

    print(
        f"Results saved to: "
        f"{result_path}"
    )


if __name__ == "__main__":
    main()
