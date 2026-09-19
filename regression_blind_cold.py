import os
import csv
import json
import pickle
import random
import time
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

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
from utils import TestbedDataset, collate, predicting, rmse, mse, pearson, spearman, ci, rm2


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def upsert_csv(path, row, key_cols, columns):
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
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(kept)


def load_raw(data_root, dataset="RNA"):
    fpath = os.path.join(data_root, dataset)
    with open(os.path.join(fpath, "ligand.json"), "r") as f:
        ligands = json.load(f, object_pairs_hook=OrderedDict)
    with open(os.path.join(fpath, "rna.json"), "r") as f:
        proteins = json.load(f, object_pairs_hook=OrderedDict)
    with open(os.path.join(fpath, "Y"), "rb") as f:
        affinity = pickle.load(f, encoding="latin1")
    return fpath, ligands, proteins, np.asarray(affinity)


def build_feature_store(data_root, dataset, ligands, proteins, rnafm_dim):
    fpath = os.path.join(data_root, dataset)
    aln_path = os.path.join(fpath, "aln")
    pconsc_path = os.path.join(fpath, "pconsc4")
    rnafm_path = os.path.join(fpath, "rnafm")

    if not (os.path.exists(aln_path) and os.path.exists(pconsc_path)):
        raise RuntimeError("Missing aln or pconsc4 directory.")

    ligand_keys = list(ligands.keys())
    ligand_values = list(ligands.values())
    protein_keys = list(proteins.keys())

    drug_df = pd.read_csv(os.path.join(data_root, "drug_embeddings.csv"))
    drug_df["ID"] = drug_df["ID"].astype(str)
    drug_emb_map = {
        row["ID"]: row.iloc[2:].values.astype(np.float32)
        for _, row in drug_df.iterrows()
    }

    smile_graph = {s: smile_to_graph(s) for s in ligand_values}
    smile_tensor = {s: label_smiles(s, 100) for s in ligand_values}
    seq_for_key = {k: clean_rna_seq(v) for k, v in proteins.items()}

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
    keys = [protein_keys[int(c)] for c in cols]

    return TestbedDataset(
        root="/tmp",
        dataset=dataset_name,
        xd=[ligand_values[int(r)] for r in rows],
        xt=[seq_cat(store["seq_for_key"][protein_keys[int(c)]]) for c in cols],
        y=np.asarray(y_values, dtype=np.float32),
        smile_graph=store["smile_graph"],
        smile_tensor=store["smile_tensor"],
        target_graph={k: store["target_graph"][k][:3] for k in keys},
        target_key=keys,
        kmer_map=kmer_map,
        rnafm_map={k: store["target_graph"][k][3] for k in keys},
        ligand_ids=np.asarray(ligand_keys)[np.asarray(rows, dtype=int)],
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
    unique_train_keys = sorted({protein_keys[int(c)] for c in train_cols})
    if len(unique_train_keys) == 0:
        raise RuntimeError("No training RNA entities available for k-mer fitting.")

    if not os.path.exists(z_path):
        train_seqs = [store["seq_for_key"][k] for k in unique_train_keys]
        fit_kmer_zscore(train_seqs, save_path=z_path)

    mean, std = load_kmer_zscore(z_path)
    return mean, std


def make_kmer_map(cols_groups, store, mean, std):
    protein_keys = store["protein_keys"]
    all_cols = np.concatenate([np.asarray(x, dtype=int) for x in cols_groups if len(x) > 0])
    needed = sorted({protein_keys[int(c)] for c in all_cols})
    return {
        str(k): zscore(seq_to_kmer_freq(store["seq_for_key"][k]), mean, std)
        for k in needed
    }


def train_regression_epoch(model, device, loader, optimizer, epoch):
    model.train()
    loss_fn = torch.nn.MSELoss()
    running = 0.0
    n = 0
    for drug, target in loader:
        drug = drug.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(drug, target)
        y = drug.y.view(-1, 1).float()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        running += float(loss.item())
        n += 1
    mean_loss = running / max(1, n)
    print(f"epoch={epoch} train_mse={mean_loss:.6f}")
    return mean_loss


MODEL_NAME = "HERBA"
SPLIT_VERSION = "exact_raw_v1"

DOUBLE_SPLIT_VERSION = "exact_raw_v2"


def split_version_for_mode(mode):
    return DOUBLE_SPLIT_VERSION if mode == "double" else SPLIT_VERSION


def _require_raw_string(value, entity_name, entity_key):

    if not isinstance(value, str):
        raise TypeError(
            f"{entity_name} {entity_key!r} must be a string for exact-raw "
            f"cold-start grouping, got {type(value).__name__}."
        )
    return value


def build_exact_raw_groups(mapping, entity_name):

    groups_by_identity = OrderedDict()

    for matrix_idx, (entity_key, raw_value) in enumerate(mapping.items()):
        identity = _require_raw_string(raw_value, entity_name, entity_key)
        groups_by_identity.setdefault(identity, []).append(int(matrix_idx))

    groups = [
        np.asarray(indices, dtype=int)
        for indices in groups_by_identity.values()
    ]

    if len(groups) == 0:
        raise RuntimeError(f"No {entity_name} groups were constructed.")

    duplicated_groups = sum(len(g) > 1 for g in groups)
    duplicated_entities = sum(len(g) for g in groups if len(g) > 1)

    print(
        f"[exact-raw grouping] {entity_name}: "
        f"matrix_entities={len(mapping)} "
        f"unique_raw_groups={len(groups)} "
        f"duplicate_groups={duplicated_groups} "
        f"entities_in_duplicate_groups={duplicated_entities}"
    )

    return groups


def group_folds(groups, n_splits, rep, seed_offset, base_seed):

    n_groups = len(groups)
    if n_groups < n_splits:
        raise RuntimeError(
            f"Not enough exact-raw groups for {n_splits}-fold CV: "
            f"n_groups={n_groups}"
        )

    group_ids = np.arange(n_groups, dtype=int)
    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(base_seed + rep * 1009 + seed_offset),
    )
    return [te.astype(int) for _, te in kf.split(group_ids)]


def expand_group_ids(groups, group_ids):
    group_ids = np.asarray(group_ids, dtype=int)
    if group_ids.size == 0:
        return np.asarray([], dtype=int)

    return np.sort(
        np.concatenate([groups[int(g)] for g in group_ids]).astype(int)
    )


def observed_pairs(affinity, allowed_rows=None, allowed_cols=None):
    rows, cols = np.where(~np.isnan(affinity))
    mask = np.ones(len(rows), dtype=bool)

    if allowed_rows is not None:
        mask &= np.isin(rows, np.asarray(allowed_rows, dtype=int))

    if allowed_cols is not None:
        mask &= np.isin(cols, np.asarray(allowed_cols, dtype=int))

    return rows[mask], cols[mask]


def split_group_ids_for_inner(group_ids, val_fraction, seed):

    group_ids = np.asarray(
        sorted(set(int(x) for x in group_ids)),
        dtype=int,
    )

    if len(group_ids) < 2:
        raise RuntimeError(
            "Not enough outer-train groups for inner cold-start validation."
        )

    tr, va = train_test_split(
        group_ids,
        test_size=val_fraction,
        random_state=int(seed),
        shuffle=True,
    )

    return np.asarray(tr, dtype=int), np.asarray(va, dtype=int)



def _entity_group_index(groups, n_entities):

    out = np.full(int(n_entities), -1, dtype=int)
    for gid, members in enumerate(groups):
        out[np.asarray(members, dtype=int)] = int(gid)
    if np.any(out < 0):
        missing = np.where(out < 0)[0]
        raise RuntimeError(
            f"Failed to assign exact-raw group ids for {len(missing)} entities."
        )
    return out


def split_double_groups_for_inner_by_pair_target(
    affinity,
    drug_groups,
    rna_groups,
    outer_train_drug_group_ids,
    outer_train_rna_group_ids,
    val_fraction,
    seed,
    n_trials=5000,
):

    outer_train_drug_group_ids = np.asarray(
        sorted(set(int(x) for x in outer_train_drug_group_ids)), dtype=int
    )
    outer_train_rna_group_ids = np.asarray(
        sorted(set(int(x) for x in outer_train_rna_group_ids)), dtype=int
    )

    if len(outer_train_drug_group_ids) < 2 or len(outer_train_rna_group_ids) < 2:
        raise RuntimeError(
            "Not enough outer-train groups for strict double-cold validation."
        )

    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError("val_fraction must be between 0 and 1.")


    row_to_group = _entity_group_index(drug_groups, affinity.shape[0])
    col_to_group = _entity_group_index(rna_groups, affinity.shape[1])

    rows_all, cols_all = np.where(~np.isnan(affinity))
    pair_dg = row_to_group[rows_all]
    pair_rg = col_to_group[cols_all]

    outer_d_set = set(int(x) for x in outer_train_drug_group_ids)
    outer_r_set = set(int(x) for x in outer_train_rna_group_ids)
    outer_mask = np.fromiter(
        (
            int(dg) in outer_d_set and int(rg) in outer_r_set
            for dg, rg in zip(pair_dg, pair_rg)
        ),
        dtype=bool,
        count=len(pair_dg),
    )
    outer_pair_dg = pair_dg[outer_mask]
    outer_pair_rg = pair_rg[outer_mask]
    n_outer_pairs = int(len(outer_pair_dg))

    if n_outer_pairs < 3:
        raise RuntimeError(
            f"Too few observed outer-train pairs for double-cold validation: "
            f"{n_outer_pairs}"
        )

    target_val_pairs = max(2, int(round(float(val_fraction) * n_outer_pairs)))


    ideal_side_fraction = float(np.sqrt(float(val_fraction)))
    rng = np.random.RandomState(int(seed))

    best = None
    best_score = None
    attempts_used = 0

    n_d_total = len(outer_train_drug_group_ids)
    n_r_total = len(outer_train_rna_group_ids)


    deterministic_fractions = [
        ideal_side_fraction,
        max(0.15, ideal_side_fraction - 0.10),
        max(0.15, ideal_side_fraction - 0.05),
        min(0.55, ideal_side_fraction + 0.05),
        min(0.55, ideal_side_fraction + 0.10),
    ]

    def evaluate_candidate(val_d_ids, val_r_ids):
        nonlocal best, best_score

        val_d_ids = np.asarray(sorted(set(int(x) for x in val_d_ids)), dtype=int)
        val_r_ids = np.asarray(sorted(set(int(x) for x in val_r_ids)), dtype=int)

        if (
            len(val_d_ids) == 0
            or len(val_r_ids) == 0
            or len(val_d_ids) >= n_d_total
            or len(val_r_ids) >= n_r_total
        ):
            return False

        val_d_set = set(int(x) for x in val_d_ids)
        val_r_set = set(int(x) for x in val_r_ids)

        val_mask = np.fromiter(
            (
                int(dg) in val_d_set and int(rg) in val_r_set
                for dg, rg in zip(outer_pair_dg, outer_pair_rg)
            ),
            dtype=bool,
            count=n_outer_pairs,
        )
        n_val_pairs = int(val_mask.sum())

        tr_d_ids = np.asarray(
            [g for g in outer_train_drug_group_ids if int(g) not in val_d_set],
            dtype=int,
        )
        tr_r_ids = np.asarray(
            [g for g in outer_train_rna_group_ids if int(g) not in val_r_set],
            dtype=int,
        )

        tr_d_set = set(int(x) for x in tr_d_ids)
        tr_r_set = set(int(x) for x in tr_r_ids)
        train_mask = np.fromiter(
            (
                int(dg) in tr_d_set and int(rg) in tr_r_set
                for dg, rg in zip(outer_pair_dg, outer_pair_rg)
            ),
            dtype=bool,
            count=n_outer_pairs,
        )
        n_train_pairs = int(train_mask.sum())

        if n_val_pairs <= 0 or n_train_pairs <= 0:
            return False


        pair_error = abs(n_val_pairs - target_val_pairs) / max(1, target_val_pairs)


        train_fraction = n_train_pairs / max(1, n_outer_pairs)
        train_penalty = max(0.0, 0.30 - train_fraction)

        score = pair_error + 0.10 * train_penalty

        if best_score is None or score < best_score:
            best_score = score
            best = (
                tr_d_ids,
                val_d_ids,
                tr_r_ids,
                val_r_ids,
                n_train_pairs,
                n_val_pairs,
            )

        # Good enough: within 5% of target or within 2 pairs.
        return abs(n_val_pairs - target_val_pairs) <= max(
            2, int(round(0.05 * target_val_pairs))
        )

    # First try several deterministic side fractions with random memberships.
    stop = False
    for frac_d in deterministic_fractions:
        if stop:
            break
        for frac_r in deterministic_fractions:
            n_val_d = min(
                n_d_total - 1,
                max(1, int(round(float(frac_d) * n_d_total))),
            )
            n_val_r = min(
                n_r_total - 1,
                max(1, int(round(float(frac_r) * n_r_total))),
            )
            # Multiple memberships for the same group counts.
            for _ in range(20):
                attempts_used += 1
                val_d = rng.choice(
                    outer_train_drug_group_ids, size=n_val_d, replace=False
                )
                val_r = rng.choice(
                    outer_train_rna_group_ids, size=n_val_r, replace=False
                )
                if evaluate_candidate(val_d, val_r):
                    stop = True
                    break
            if stop:
                break


    if not stop:
        low = max(0.12, ideal_side_fraction * 0.55)
        high = min(0.60, ideal_side_fraction * 1.65)
        for _ in range(int(n_trials)):
            attempts_used += 1

            frac_d = float(np.clip(rng.normal(ideal_side_fraction, 0.09), low, high))
            frac_r = float(np.clip(rng.normal(ideal_side_fraction, 0.09), low, high))

            n_val_d = min(
                n_d_total - 1,
                max(1, int(round(frac_d * n_d_total))),
            )
            n_val_r = min(
                n_r_total - 1,
                max(1, int(round(frac_r * n_r_total))),
            )

            val_d = rng.choice(
                outer_train_drug_group_ids, size=n_val_d, replace=False
            )
            val_r = rng.choice(
                outer_train_rna_group_ids, size=n_val_r, replace=False
            )

            if evaluate_candidate(val_d, val_r):
                break

    if best is None:
        raise RuntimeError(
            "Could not construct a non-empty strict double-cold inner validation "
            f"split after {attempts_used} candidate searches."
        )

    (
        inner_train_drug_group_ids,
        val_drug_group_ids,
        inner_train_rna_group_ids,
        val_rna_group_ids,
        selected_train_pairs,
        selected_val_pairs,
    ) = best


    min_acceptable_val_pairs = min(
        target_val_pairs,
        max(5, int(round(0.50 * target_val_pairs))),
    )
    if selected_val_pairs < min_acceptable_val_pairs:
        raise RuntimeError(
            "Strict double-cold validation search found too few validation pairs: "
            f"outer_pairs={n_outer_pairs}, target={target_val_pairs}, "
            f"best={selected_val_pairs}, required>={min_acceptable_val_pairs}. "
            "Increase --double-search-trials if needed."
        )

    print(
        "[double-inner-search] "
        f"outer_train_pairs={n_outer_pairs} "
        f"target_val_pairs={target_val_pairs} "
        f"selected_val_pairs={selected_val_pairs} "
        f"selected_train_pairs={selected_train_pairs} "
        f"drug_groups train/val="
        f"{len(inner_train_drug_group_ids)}/{len(val_drug_group_ids)} "
        f"rna_groups train/val="
        f"{len(inner_train_rna_group_ids)}/{len(val_rna_group_ids)} "
        f"attempts={attempts_used}"
    )

    return (
        np.asarray(inner_train_drug_group_ids, dtype=int),
        np.asarray(val_drug_group_ids, dtype=int),
        np.asarray(inner_train_rna_group_ids, dtype=int),
        np.asarray(val_rna_group_ids, dtype=int),
    )


def raw_identity_set(mapping_values, matrix_indices):

    return {
        mapping_values[int(i)]
        for i in set(int(x) for x in matrix_indices)
    }


def overlap_count(a, b):
    return int(len(set(a) & set(b)))


def assert_disjoint_three(train_set, val_set, test_set, label):
    tv = overlap_count(train_set, val_set)
    tt = overlap_count(train_set, test_set)
    vt = overlap_count(val_set, test_set)

    if tv != 0 or tt != 0 or vt != 0:
        raise AssertionError(
            f"{label} raw-identity leakage detected: "
            f"train-val={tv}, train-test={tt}, val-test={vt}"
        )


def create_cold_dataset(
    data_root,
    dataset,
    mode,
    rep,
    fold,
    n_splits,
    val_fraction,
    rnafm_dim,
    base_seed,
    double_search_trials=5000,
):
    fpath, ligands, proteins, affinity = load_raw(data_root, dataset)
    split_version = split_version_for_mode(mode)

    ligand_keys = list(ligands.keys())
    ligand_values = list(ligands.values())
    protein_keys = list(proteins.keys())
    protein_values = list(proteins.values())

    expected_shape = (len(ligand_values), len(protein_values))
    if tuple(affinity.shape) != expected_shape:
        raise RuntimeError(
            "Y shape does not match ligand.json/rna.json ordering: "
            f"Y={tuple(affinity.shape)}, "
            f"expected=({len(ligand_values)}, {len(protein_values)})."
        )


    drug_groups = build_exact_raw_groups(ligands, "Drug/SMILES")
    rna_groups = build_exact_raw_groups(proteins, "RNA/sequence")

    drug_folds = group_folds(
        drug_groups, n_splits, rep, seed_offset=101, base_seed=base_seed
    )
    rna_folds = group_folds(
        rna_groups, n_splits, rep, seed_offset=202, base_seed=base_seed
    )

    all_drug_group_ids = np.arange(len(drug_groups), dtype=int)
    all_rna_group_ids = np.arange(len(rna_groups), dtype=int)

    held_drug_group_ids = np.asarray(drug_folds[fold], dtype=int)
    held_rna_group_ids = np.asarray(rna_folds[fold], dtype=int)

    seed = int(base_seed + rep * 1000 + fold)

    if mode == "rna":

        outer_train_rna_group_ids = np.setdiff1d(
            all_rna_group_ids,
            held_rna_group_ids,
        )

        inner_train_rna_group_ids, val_rna_group_ids = (
            split_group_ids_for_inner(
                outer_train_rna_group_ids,
                val_fraction,
                seed,
            )
        )

        inner_train_rnas = expand_group_ids(
            rna_groups, inner_train_rna_group_ids
        )
        val_rnas = expand_group_ids(
            rna_groups, val_rna_group_ids
        )
        held_rnas = expand_group_ids(
            rna_groups, held_rna_group_ids
        )

        tr_r, tr_c = observed_pairs(
            affinity,
            allowed_cols=inner_train_rnas,
        )
        va_r, va_c = observed_pairs(
            affinity,
            allowed_cols=val_rnas,
        )
        te_r, te_c = observed_pairs(
            affinity,
            allowed_cols=held_rnas,
        )

        assert set(tr_c).isdisjoint(set(va_c))
        assert set(tr_c).isdisjoint(set(te_c))
        assert set(va_c).isdisjoint(set(te_c))

    elif mode == "drug":

        outer_train_drug_group_ids = np.setdiff1d(
            all_drug_group_ids,
            held_drug_group_ids,
        )

        inner_train_drug_group_ids, val_drug_group_ids = (
            split_group_ids_for_inner(
                outer_train_drug_group_ids,
                val_fraction,
                seed,
            )
        )

        inner_train_drugs = expand_group_ids(
            drug_groups, inner_train_drug_group_ids
        )
        val_drugs = expand_group_ids(
            drug_groups, val_drug_group_ids
        )
        held_drugs = expand_group_ids(
            drug_groups, held_drug_group_ids
        )

        tr_r, tr_c = observed_pairs(
            affinity,
            allowed_rows=inner_train_drugs,
        )
        va_r, va_c = observed_pairs(
            affinity,
            allowed_rows=val_drugs,
        )
        te_r, te_c = observed_pairs(
            affinity,
            allowed_rows=held_drugs,
        )

        assert set(tr_r).isdisjoint(set(va_r))
        assert set(tr_r).isdisjoint(set(te_r))
        assert set(va_r).isdisjoint(set(te_r))

    elif mode == "double":

        outer_train_drug_group_ids = np.setdiff1d(
            all_drug_group_ids,
            held_drug_group_ids,
        )
        outer_train_rna_group_ids = np.setdiff1d(
            all_rna_group_ids,
            held_rna_group_ids,
        )

        (
            inner_train_drug_group_ids,
            val_drug_group_ids,
            inner_train_rna_group_ids,
            val_rna_group_ids,
        ) = split_double_groups_for_inner_by_pair_target(
            affinity=affinity,
            drug_groups=drug_groups,
            rna_groups=rna_groups,
            outer_train_drug_group_ids=outer_train_drug_group_ids,
            outer_train_rna_group_ids=outer_train_rna_group_ids,
            val_fraction=val_fraction,
            seed=seed + 7919,
            n_trials=double_search_trials,
        )

        inner_train_drugs = expand_group_ids(
            drug_groups, inner_train_drug_group_ids
        )
        val_drugs = expand_group_ids(
            drug_groups, val_drug_group_ids
        )
        held_drugs = expand_group_ids(
            drug_groups, held_drug_group_ids
        )

        inner_train_rnas = expand_group_ids(
            rna_groups, inner_train_rna_group_ids
        )
        val_rnas = expand_group_ids(
            rna_groups, val_rna_group_ids
        )
        held_rnas = expand_group_ids(
            rna_groups, held_rna_group_ids
        )


        tr_r, tr_c = observed_pairs(
            affinity,
            allowed_rows=inner_train_drugs,
            allowed_cols=inner_train_rnas,
        )
        va_r, va_c = observed_pairs(
            affinity,
            allowed_rows=val_drugs,
            allowed_cols=val_rnas,
        )
        te_r, te_c = observed_pairs(
            affinity,
            allowed_rows=held_drugs,
            allowed_cols=held_rnas,
        )

        # Matrix IDs are disjoint on BOTH sides.
        assert set(tr_r).isdisjoint(set(va_r))
        assert set(tr_r).isdisjoint(set(te_r))
        assert set(va_r).isdisjoint(set(te_r))

        assert set(tr_c).isdisjoint(set(va_c))
        assert set(tr_c).isdisjoint(set(te_c))
        assert set(va_c).isdisjoint(set(te_c))

    else:
        raise ValueError(mode)

    if min(len(tr_r), len(va_r), len(te_r)) == 0:
        raise RuntimeError(
            f"Empty split in mode={mode}, rep={rep}, fold={fold}: "
            f"train={len(tr_r)}, val={len(va_r)}, test={len(te_r)}"
        )


    train_rna_raw = raw_identity_set(protein_values, tr_c)
    val_rna_raw = raw_identity_set(protein_values, va_c)
    test_rna_raw = raw_identity_set(protein_values, te_c)

    train_drug_raw = raw_identity_set(ligand_values, tr_r)
    val_drug_raw = raw_identity_set(ligand_values, va_r)
    test_drug_raw = raw_identity_set(ligand_values, te_r)

    rna_tv_overlap = overlap_count(train_rna_raw, val_rna_raw)
    rna_tt_overlap = overlap_count(train_rna_raw, test_rna_raw)
    rna_vt_overlap = overlap_count(val_rna_raw, test_rna_raw)

    drug_tv_overlap = overlap_count(train_drug_raw, val_drug_raw)
    drug_tt_overlap = overlap_count(train_drug_raw, test_drug_raw)
    drug_vt_overlap = overlap_count(val_drug_raw, test_drug_raw)

    if mode in ("rna", "double"):
        assert_disjoint_three(
            train_rna_raw,
            val_rna_raw,
            test_rna_raw,
            "RNA sequence",
        )

    if mode in ("drug", "double"):
        assert_disjoint_three(
            train_drug_raw,
            val_drug_raw,
            test_drug_raw,
            "Drug SMILES",
        )

    print(
        f"[cold-exact-raw-{mode}] rep={rep} fold={fold}/{n_splits}: "
        f"pairs train={len(tr_r)} val={len(va_r)} test={len(te_r)}"
    )
    print(
        "  matrix RNA IDs train/val/test="
        f"{len(set(tr_c))}/{len(set(va_c))}/{len(set(te_c))}; "
        "raw RNA groups train/val/test="
        f"{len(train_rna_raw)}/{len(val_rna_raw)}/{len(test_rna_raw)}"
    )
    print(
        "  matrix Drug IDs train/val/test="
        f"{len(set(tr_r))}/{len(set(va_r))}/{len(set(te_r))}; "
        "raw SMILES groups train/val/test="
        f"{len(train_drug_raw)}/{len(val_drug_raw)}/{len(test_drug_raw)}"
    )
    print(
        "  raw RNA overlap train-val/train-test/val-test="
        f"{rna_tv_overlap}/{rna_tt_overlap}/{rna_vt_overlap}"
    )
    print(
        "  raw SMILES overlap train-val/train-test/val-test="
        f"{drug_tv_overlap}/{drug_tt_overlap}/{drug_vt_overlap}"
    )

    store = build_feature_store(
        data_root,
        dataset,
        ligands,
        proteins,
        rnafm_dim,
    )

    # New filename prevents reuse of k-mer statistics from the old split.
    z_path = os.path.join(
        fpath,
        f"kmer_z_{split_version}_{mode}_{n_splits}fold_r{rep}_f{fold}.npz",
    )

    km_mean, km_std = fit_kmer_from_train_cols(
        data_root,
        dataset,
        tr_c,
        store,
        z_path,
    )
    kmer_map = make_kmer_map(
        [tr_c, va_c, te_c],
        store,
        km_mean,
        km_std,
    )

    meta = {
        "n_train_pairs": int(len(tr_r)),
        "n_val_pairs": int(len(va_r)),
        "n_test_pairs": int(len(te_r)),


        "n_train_rna": int(len(set(tr_c))),
        "n_val_rna": int(len(set(va_c))),
        "n_test_rna": int(len(set(te_c))),
        "n_train_drug": int(len(set(tr_r))),
        "n_val_drug": int(len(set(va_r))),
        "n_test_drug": int(len(set(te_r))),


        "n_train_rna_raw_groups": int(len(train_rna_raw)),
        "n_val_rna_raw_groups": int(len(val_rna_raw)),
        "n_test_rna_raw_groups": int(len(test_rna_raw)),
        "n_train_drug_raw_groups": int(len(train_drug_raw)),
        "n_val_drug_raw_groups": int(len(val_drug_raw)),
        "n_test_drug_raw_groups": int(len(test_drug_raw)),


        "rna_train_val_raw_overlap": rna_tv_overlap,
        "rna_train_test_raw_overlap": rna_tt_overlap,
        "rna_val_test_raw_overlap": rna_vt_overlap,
        "drug_train_val_raw_overlap": drug_tv_overlap,
        "drug_train_test_raw_overlap": drug_tt_overlap,
        "drug_val_test_raw_overlap": drug_vt_overlap,
    }

    return (
        make_dataset_from_pairs(
            f"{dataset}_{split_version}_{mode}_r{rep}_f{fold}_train",
            tr_r,
            tr_c,
            affinity[tr_r, tr_c],
            store,
            kmer_map,
            rnafm_dim,
        ),
        make_dataset_from_pairs(
            f"{dataset}_{split_version}_{mode}_r{rep}_f{fold}_val",
            va_r,
            va_c,
            affinity[va_r, va_c],
            store,
            kmer_map,
            rnafm_dim,
        ),
        make_dataset_from_pairs(
            f"{dataset}_{split_version}_{mode}_r{rep}_f{fold}_test",
            te_r,
            te_c,
            affinity[te_r, te_c],
            store,
            kmer_map,
            rnafm_dim,
        ),
        meta,
    )


def main():
    from pathlib import Path

    project_root = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(
        description=(
            "HERBA exact-identity cold-start regression. "
            "Modes: RNA cold-start, small-molecule cold-start, "
            "and dual cold-start."
        )
    )

    ap.add_argument(
        "--action",
        choices=["train", "test"],
        default="train",
    )
    ap.add_argument(
        "--mode",
        choices=["rna", "drug", "double"],
        required=True,
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of outer folds. The released benchmark uses 5.",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=1500,
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=100,
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=128,
    )
    ap.add_argument(
        "--batch-test",
        type=int,
        default=128,
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    ap.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
    )
    ap.add_argument(
        "--double-search-trials",
        type=int,
        default=5000,
        help=(
            "Candidate group splits searched for double-cold "
            "inner validation."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    ap.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    ap.add_argument(
        "--rnafm-dim",
        type=int,
        default=640,
    )
    ap.add_argument(
        "--data-root",
        default=os.environ.get(
            "HERBA_DATA_DIR",
            str(project_root / "data"),
        ),
        help="HERBA data directory. Default: ./data",
    )
    ap.add_argument(
        "--checkpoint-root",
        default=str(project_root / "Checkpoint"),
        help=(
            "Root directory containing released HERBA checkpoint bundles."
        ),
    )
    ap.add_argument(
        "--rebuild-cache",
        action="store_true",
        help=(
            "Rebuild train/val/test cache files during training even if "
            "they already exist."
        ),
    )

    args = ap.parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be >= 2")

    split_version = split_version_for_mode(args.mode)

    task_name_map = {
        "rna": "blind_RNA",
        "drug": "blind_small_molecule",
        "double": "blind_double",
    }
    task_name = task_name_map[args.mode]

    device = torch.device(
        f"cuda:{args.gpu}"
        if torch.cuda.is_available()
        else "cpu"
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

    meta_columns = [
        "n_train_pairs",
        "n_val_pairs",
        "n_test_pairs",
        "n_train_rna",
        "n_val_rna",
        "n_test_rna",
        "n_train_drug",
        "n_val_drug",
        "n_test_drug",
        "n_train_rna_raw_groups",
        "n_val_rna_raw_groups",
        "n_test_rna_raw_groups",
        "n_train_drug_raw_groups",
        "n_val_drug_raw_groups",
        "n_test_drug_raw_groups",
        "rna_train_val_raw_overlap",
        "rna_train_test_raw_overlap",
        "rna_val_test_raw_overlap",
        "drug_train_val_raw_overlap",
        "drug_train_test_raw_overlap",
        "drug_val_test_raw_overlap",
    ]

    columns = [
        "dataset",
        "mode",
        "rep",
        "fold",
        "flat_fold",
        "model",
        "best_epoch",
        "val_mse",
        *meta_columns,
        "rmse",
        "mse",
        "pearson",
        "spearman",
        "ci",
        "rm2",
    ]

    print("=" * 80)
    print("HERBA exact-identity cold-start regression")
    print(f"Action          : {args.action}")
    print(f"Mode            : {args.mode}")
    print(f"Task directory  : {task_dir}")
    print(f"Dataset cache   : {cache_dir}")
    print(f"Folds           : {args.folds}")
    print(f"Repeats         : {args.repeats}")
    print(f"Device          : {device}")
    print(f"Data root       : {args.data_root}")
    print("=" * 80)

    for rep in range(args.repeats):
        for fold in range(args.folds):
            fold_seed = (
                args.seed
                + rep * 1000
                + fold
            )
            seed_everything(fold_seed)


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
            meta_path = (
                cache_dir
                / f"f{flat_fold}_meta.json"
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


            meta = {
                key: ""
                for key in meta_columns
            }

            if args.action == "test":
                missing = [
                    str(path)
                    for path in (
                        test_cache,
                        ckpt_path,
                    )
                    if not path.is_file()
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

                if meta_path.is_file():
                    with open(meta_path, "r") as f:
                        loaded_meta = json.load(f)
                    meta.update(loaded_meta)

            else:
                use_existing_cache = (
                    not args.rebuild_cache
                    and all(
                        path.is_file()
                        for path in cache_paths
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
                            path,
                            map_location="cpu",
                        )
                        for path in cache_paths
                    ]

                    if meta_path.is_file():
                        with open(meta_path, "r") as f:
                            loaded_meta = json.load(f)
                        meta.update(loaded_meta)

                else:
                    print(
                        f"[cache] building f{flat_fold} "
                        f"(rep={rep}, fold={fold})"
                    )

                    (
                        train_data,
                        val_data,
                        test_data,
                        built_meta,
                    ) = create_cold_dataset(
                        args.data_root,
                        "RNA",
                        args.mode,
                        rep,
                        fold,
                        args.folds,
                        args.val_fraction,
                        args.rnafm_dim,
                        args.seed,
                        args.double_search_trials,
                    )

                    meta.update(built_meta)

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
                        built_meta,
                        meta_path,
                    )

            if args.action == "train":
                train_loader = torch.utils.data.DataLoader(
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

                val_loader = torch.utils.data.DataLoader(
                    val_data,
                    batch_size=args.batch_test,
                    shuffle=False,
                    collate_fn=collate,
                )

                model = HERBA(
                    rnafm_dim=args.rnafm_dim
                ).to(device)

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=args.lr,
                )

                best_val = float("inf")
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
                                "best_val_mse": best_val,
                                "model_state_dict": (
                                    model.state_dict()
                                ),
                                "model": MODEL_NAME,
                                "task": task_name,
                                "mode": args.mode,
                                "split_version": split_version,
                                "n_splits": args.folds,
                                "repeat": rep,
                                "fold": fold,
                                "flat_fold": flat_fold,
                                "seed": fold_seed,
                            },
                            ckpt_path,
                        )

                        print(
                            f"[VAL+] mode={args.mode} "
                            f"rep={rep} "
                            f"fold={fold} "
                            f"flat=f{flat_fold} "
                            f"epoch={epoch} "
                            f"mse={vmse:.6f}"
                        )

                    else:
                        stale += 1

                    if stale >= args.patience:
                        print(
                            f"[EARLY] mode={args.mode} "
                            f"rep={rep} "
                            f"fold={fold} "
                            f"flat=f{flat_fold} "
                            f"best_epoch={best_epoch}"
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
                    f"Checkpoint not found: {ckpt_path}"
                )

            checkpoint = load_torch(
                ckpt_path,
                map_location=device,
            )

            model = HERBA(
                rnafm_dim=args.rnafm_dim
            ).to(device)

            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )

            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=args.batch_test,
                shuffle=False,
                collate_fn=collate,
            )

            y, p = predicting(
                model,
                device,
                test_loader,
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
                "rmse": float(rmse(y, p)),
                "mse": float(mse(y, p)),
                "pearson": float(
                    pearson(y, p)
                ),
                "spearman": float(
                    spearman(y, p)
                ),
                "ci": float(ci(y, p)),
                "rm2": float(rm2(y, p)),
            }

            upsert_csv(
                result_path,
                row,
                [
                    "dataset",
                    "mode",
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

            if train_data is not None:
                del train_data
            if val_data is not None:
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
