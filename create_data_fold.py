import os
import json
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from utils import TestbedDataset
from kmer import (
    seq_to_kmer_freq,
    fit_kmer_zscore,
    load_kmer_zscore,
    zscore,
)


def safe_mol_from_smiles(smi: str):
    if not isinstance(smi, str) or smi.strip() == "":
        return None
    parts = smi.strip().split(".")
    parts.sort(key=len, reverse=True)
    for cand in parts:
        mol = Chem.MolFromSmiles(cand)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                return mol
            except Exception:
                continue
    return None


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C","N","O","S","F","Si","P","Cl","Br","Mg","Na","Ca",
                "Fe","As","Al","I","B","V","K","Tl","Yb","Sb","Sn","Ag",
                "Pd","Co","Se","Ti","Zn","H","Li","Ge","Cu","Au","Ni","Cd",
                "In","Mn","Zr","Cr","Pt","Hg","Pb","Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
        + [atom.GetIsAromatic()]
    )


def smile_to_graph(smile):
    mol = safe_mol_from_smiles(smile)
    if mol is None:
        mol = Chem.MolFromSmiles("C")

    features = [
        np.asarray(atom_features(atom), dtype=np.float32)
        for atom in mol.GetAtoms()
    ]
    c_size = len(features)

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) == 0:
        edge_index = [[i, i] for i in range(c_size)]

    return c_size, features, edge_index


def label_smiles(line, MAX_SMI_LEN=100):
    charisosmiset = {
        "#":29,"%":30,")":31,"(":1,"+":32,"-":33,"/":34,".":2,
        "1":35,"0":3,"3":36,"2":4,"5":37,"4":5,"7":38,"6":6,
        "9":39,"8":7,"=":40,"A":41,"@":8,"C":42,"B":9,"E":43,
        "D":10,"G":44,"F":11,"I":45,"H":12,"K":46,"M":47,"L":13,
        "O":48,"N":14,"P":15,"S":49,"R":16,"U":50,"T":17,"W":51,
        "V":18,"Y":52,"[":53,"Z":19,"]":54,"\\":20,"a":55,"c":56,
        "b":21,"e":57,"d":22,"g":58,"f":23,"i":59,"h":24,"m":60,
        "l":25,"o":61,"n":26,"s":62,"r":27,"u":63,"t":28,"y":64,
    }
    x = np.zeros(MAX_SMI_LEN, dtype=np.int64)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        x[i] = charisosmiset.get(ch, 0)
    return x


pro_res_table = [
    "A","U","G","C","D","E","F","H","I","K","L","M",
    "N","P","Q","R","S","V","W","Y","X",
]

RNA_PROP = {
    "A": [1,0,1,0,1,135.13,1,1,2,5,0,-12.0],
    "C": [0,1,1,1,1,111.10,1,2,1,3,1,-9.0],
    "G": [1,0,1,1,1,151.13,2,1,2,5,1,-14.0],
    "U": [0,1,1,1,0,112.09,1,2,1,2,2,-8.0],
    "N": [0.5,0.5,1,0.75,0.75,127.36,1.25,1.5,1.5,3.75,1,-10.0],
}

USE_RNA12_MASK = np.array(
    [True, True, True, False, False, False, False, False, False, False, False, False],
    dtype=bool,
)
USE_RNA12_SCALE = "zscore"


def _read_ref_keep_idx(aln_file):
    with open(aln_file, "r") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    ref_aln = None
    for line in lines:
        if not line.startswith(">"):
            ref_aln = line
            break

    if ref_aln is None:
        raise ValueError(f"{aln_file}: no reference sequence found")

    keep_idx = [i for i, ch in enumerate(ref_aln) if ch != "-"]
    return keep_idx, ref_aln, lines


def _scale_rna12(mat, method, start_col=5):
    out = np.asarray(mat, dtype=float).copy()
    if out.shape[0] == 0 or method == "none":
        return out

    if method == "minmax":
        for j in range(start_col, out.shape[1]):
            col = out[:, j]
            cmin = np.min(col)
            cmax = np.max(col)
            out[:, j] = 0.0 if cmax <= cmin else (col - cmin) / (cmax - cmin)
        return out

    if method == "zscore":
        for j in range(start_col, out.shape[1]):
            col = out[:, j]
            mu = np.mean(col)
            std = np.std(col)
            out[:, j] = 0.0 if std == 0 else (col - mu) / std
        return out

    return out


def rna12_vector(base):
    return np.asarray(RNA_PROP.get(base, RNA_PROP["N"]), dtype=float)


def clean_rna_seq(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace("T", "U")
    return "".join(ch for ch in seq if ch in set("ACGUN"))


def PSSM_calculation(aln_file, pro_seq, clip_len=None):
    vocab = pro_res_table
    aa_to_idx = {aa: i for i, aa in enumerate(vocab)}
    keep_idx, _, raw_lines = _read_ref_keep_idx(aln_file)

    length = len(keep_idx)
    if clip_len is not None and clip_len < length:
        keep_idx = keep_idx[:clip_len]
        length = clip_len

    if len(pro_seq) < length:
        print(
            f"[WARN] PSSM L={length} > pro_seq L={len(pro_seq)} "
            f"-> truncate to {len(pro_seq)}"
        )
        keep_idx = keep_idx[:len(pro_seq)]
        length = len(pro_seq)

    pfm = np.zeros((len(vocab), length), dtype=np.float32)
    valid_rows = 0

    for line in raw_lines:
        if line.startswith(">"):
            continue
        if length == 0 or len(line) < keep_idx[-1] + 1:
            continue

        col = 0
        for j in keep_idx:
            ch = line[j].upper().replace("T", "U")
            if ch not in aa_to_idx:
                ch = "X"
            pfm[aa_to_idx[ch], col] += 1.0
            col += 1
        valid_rows += 1

    if valid_rows == 0:
        print(f"[WARN] {aln_file}: zero valid alignment rows")

    pseudocount = 0.8
    k = float(len(vocab))
    return (pfm + pseudocount / k) / (float(valid_rows) + pseudocount)


def seq_feature(pro_seq):
    onehot_dim = len(pro_res_table)
    length = len(pro_seq)

    pro_hot = np.zeros((length, onehot_dim), dtype=float)
    rna12 = np.zeros((length, 12), dtype=float)

    for i, ch in enumerate(pro_seq):
        ch = ch.upper().replace("T", "U")
        ch_use = ch if ch in pro_res_table else "X"
        pro_hot[i] = one_of_k_encoding(ch_use, pro_res_table)
        base = ch if ch in ("A", "C", "G", "U", "N") else "N"
        rna12[i] = rna12_vector(base)

    rna12 = rna12 * USE_RNA12_MASK[None, :].astype(float)
    rna12 = _scale_rna12(rna12, method=USE_RNA12_SCALE, start_col=5)
    return np.concatenate([pro_hot, rna12], axis=1)


def target_feature(aln_file, pro_seq, clip_len=None):
    pro_seq = clean_rna_seq(pro_seq)

    if clip_len is not None:
        if len(pro_seq) < clip_len:
            pro_seq += "N" * (clip_len - len(pro_seq))
        elif len(pro_seq) > clip_len:
            pro_seq = pro_seq[:clip_len]

    pssm = PSSM_calculation(aln_file, pro_seq, clip_len=clip_len)
    length = pssm.shape[1]
    other = seq_feature(pro_seq[:length])

    if other.shape[0] < length:
        pad = np.zeros(
            (length - other.shape[0], other.shape[1]),
            dtype=other.dtype,
        )
        other = np.concatenate([other, pad], axis=0)
    elif other.shape[0] > length:
        other = other[:length]

    feat = np.concatenate([pssm.T, other], axis=1).astype(np.float32)

    if clip_len is not None and feat.shape[0] != clip_len:
        if feat.shape[0] < clip_len:
            pad = np.zeros(
                (clip_len - feat.shape[0], feat.shape[1]),
                dtype=np.float32,
            )
            feat = np.concatenate([feat, pad], axis=0)
        else:
            feat = feat[:clip_len]

    if feat.shape[1] != 54:
        raise ValueError(
            f"target_feature dim error: got {feat.shape[1]}, expected 54"
        )
    return feat


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {value: i + 1 for i, value in enumerate(seq_voc)}
max_seq_len = 1000


def seq_cat(prot):
    prot = clean_rna_seq(prot)
    x = np.zeros(max_seq_len, dtype=np.int64)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict.get(ch, 0)
    return x


def _load_and_align_rnafm(fm_path, length, rnafm_dim_default=640):
    if os.path.exists(fm_path):
        fm = np.load(fm_path).astype(np.float32)
        if fm.ndim != 2:
            raise ValueError(
                f"{fm_path}: RNA-FM ndim={fm.ndim}; expected 2"
            )
        if fm.shape[1] != rnafm_dim_default:
            raise ValueError(
                f"{fm_path}: RNA-FM dim={fm.shape[1]}; "
                f"expected {rnafm_dim_default}"
            )

        if fm.shape[0] >= length:
            fm = fm[:length]
        else:
            tmp = np.zeros(
                (length, fm.shape[1]), dtype=np.float32
            )
            tmp[:fm.shape[0]] = fm
            fm = tmp
    else:
        fm = np.zeros(
            (length, rnafm_dim_default), dtype=np.float32
        )

    return fm, fm.mean(axis=0).astype(np.float32)


def target_to_graph(
    target_key,
    target_sequence,
    contact_dir,
    aln_dir,
    rnafm_dir=None,
    rnafm_dim_default=640,
):
    contact_file = os.path.join(contact_dir, target_key + ".npy")
    contact_map = np.load(contact_file)
    contact_map = contact_map + np.eye(
        contact_map.shape[0], dtype=contact_map.dtype
    )
    length = contact_map.shape[0]

    row, col = np.where(contact_map >= 0.5)
    target_edge_index = np.asarray(
        [[i, j] for i, j in zip(row, col)],
        dtype=np.int64,
    )

    aln_file = os.path.join(aln_dir, target_key + ".aln")
    raw_feat = target_feature(
        aln_file, target_sequence, clip_len=length
    )

    fm_path = (
        os.path.join(rnafm_dir, target_key + ".npy")
        if rnafm_dir is not None else ""
    )
    fm, rnafm_global = _load_and_align_rnafm(
        fm_path, length, rnafm_dim_default
    )

    node_feat = np.concatenate(
        [raw_feat, fm], axis=1
    ).astype(np.float32)

    expected = 54 + rnafm_dim_default
    if node_feat.shape[1] != expected:
        raise ValueError(
            f"TargetData.x dim={node_feat.shape[1]}, expected={expected}"
        )

    return length, node_feat, target_edge_index, rnafm_global


def get_or_make_folds_for_pairs(fpath, affinity, k=5, seed=1, rep=0):
    folds_dir = os.path.join(fpath, "folds")
    os.makedirs(folds_dir, exist_ok=True)

    folds_file = os.path.join(
        folds_dir, f"pair_kfold_{k}_rep{rep}.json"
    )
    rows_all, cols_all = np.where(~np.isnan(affinity))

    if os.path.exists(folds_file):
        with open(folds_file, "r") as handle:
            folds = json.load(handle)
        folds = [[int(x) for x in fold] for fold in folds]
        return rows_all, cols_all, folds

    kfold = KFold(
        n_splits=k,
        shuffle=True,
        random_state=seed + rep,
    )
    idx_all = np.arange(len(rows_all))
    folds = []

    for _, test_idx in kfold.split(idx_all):
        folds.append(test_idx.tolist())

    with open(folds_file, "w") as handle:
        json.dump(folds, handle)

    return rows_all, cols_all, folds


def split_outer_train_validation(
    outer_train_idx,
    rows_all,
    cols_all,
    affinity,
    fold,
    rep,
    val_fraction=0.10,
    seed=2026,
):
    outer_train_idx = np.asarray(outer_train_idx, dtype=int)
    y_outer = affinity[
        rows_all[outer_train_idx],
        cols_all[outer_train_idx],
    ]

    rng_seed = int(seed + rep * 1000 + fold)
    stratify = None

    try:
        n_bins = min(
            10, max(2, len(outer_train_idx) // 50)
        )
        bins = pd.qcut(
            y_outer,
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        counts = pd.Series(bins).value_counts()
        if len(counts) >= 2 and counts.min() >= 2:
            stratify = np.asarray(bins)
    except (ValueError, TypeError):
        stratify = None

    inner_train_idx, val_idx = train_test_split(
        outer_train_idx,
        test_size=val_fraction,
        random_state=rng_seed,
        shuffle=True,
        stratify=stratify,
    )
    return (
        np.asarray(inner_train_idx, dtype=int),
        np.asarray(val_idx, dtype=int),
    )


def create_dataset(dataset, fold=0, rep=0, val_fraction=0.10):
    # Place each variant directly under HERBA/<variant>;
    # they all share HERBA/data.
    data_root = os.environ.get("HERBA_DATA_DIR", "../data")
    fpath = os.path.join(data_root, dataset)

    with open(os.path.join(fpath, "ligand.json"), "r") as handle:
        ligands = json.load(handle, object_pairs_hook=OrderedDict)
    with open(os.path.join(fpath, "rna.json"), "r") as handle:
        proteins = json.load(handle, object_pairs_hook=OrderedDict)
    with open(os.path.join(fpath, "Y"), "rb") as handle:
        affinity = pickle.load(handle, encoding="latin1")

    affinity = np.asarray(affinity)
    aln_path = os.path.join(fpath, "aln")
    pconsc_path = os.path.join(fpath, "pconsc4")

    if not (
        os.path.exists(aln_path)
        and os.path.exists(pconsc_path)
    ):
        raise RuntimeError("Missing aln or pconsc4 directory.")

    rows_all, cols_all, folds = get_or_make_folds_for_pairs(
        fpath, affinity, k=5, seed=1, rep=rep
    )

    test_pair_idx = np.asarray(folds[fold], dtype=int)
    outer_train_pair_idx = np.concatenate(
        [
            np.asarray(one_fold, dtype=int)
            for i, one_fold in enumerate(folds)
            if i != fold
        ],
        axis=0,
    )

    train_pair_idx, val_pair_idx = (
        split_outer_train_validation(
            outer_train_pair_idx,
            rows_all,
            cols_all,
            affinity,
            fold=fold,
            rep=rep,
            val_fraction=val_fraction,
        )
    )

    train_rows = rows_all[train_pair_idx]
    train_cols = cols_all[train_pair_idx]
    val_rows = rows_all[val_pair_idx]
    val_cols = cols_all[val_pair_idx]
    test_rows = rows_all[test_pair_idx]
    test_cols = cols_all[test_pair_idx]

    print(
        f"[split] rep={rep} outer_fold={fold}: "
        f"train={len(train_pair_idx)}, "
        f"val={len(val_pair_idx)}, "
        f"test={len(test_pair_idx)}"
    )

    ligand_keys = list(ligands.keys())
    ligand_values = list(ligands.values())
    protein_keys = list(proteins.keys())

    drug_df = pd.read_csv(
        os.path.join(data_root, "drug_embeddings.csv")
    )
    drug_df["ID"] = drug_df["ID"].astype(str)
    drug_emb_map = {
        row["ID"]: row.iloc[2:].values.astype(np.float32)
        for _, row in drug_df.iterrows()
    }

    smile_graph = {
        smile: smile_to_graph(smile)
        for smile in ligand_values
    }
    smile_tensor = {
        smile: label_smiles(smile, 100)
        for smile in ligand_values
    }

    seq_for_key = {
        key: clean_rna_seq(value)
        for key, value in proteins.items()
    }

    rnafm_path = os.path.join(fpath, "rnafm")
    rnafm_dim = int(
        os.environ.get("RNAFM_DIM", 640)
    )

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

    z_path = os.path.join(
        fpath,
        f"kmer_z_nested_v1_r{rep}_f{fold}.npz",
    )

    unique_train_keys = sorted(
        {protein_keys[col] for col in train_cols}
    )
    train_seqs = [
        seq_for_key[key]
        for key in unique_train_keys
    ]

    if not os.path.exists(z_path):
        fit_kmer_zscore(
            train_seqs,
            save_path=z_path,
        )

    km_mean, km_std = load_kmer_zscore(z_path)

    all_needed_keys = sorted(
        {
            protein_keys[col]
            for col in np.concatenate(
                [train_cols, val_cols, test_cols]
            )
        }
    )
    kmer_map = {
        str(key): zscore(
            seq_to_kmer_freq(seq_for_key[key]),
            km_mean,
            km_std,
        )
        for key in all_needed_keys
    }

    ligand_ids = np.asarray(ligand_keys)

    def make_split(name, rows, cols):
        keys = [protein_keys[col] for col in cols]
        return TestbedDataset(
            root=data_root,
            dataset=f"{dataset}_r{rep}_f{fold}_{name}",
            xd=[ligand_values[row] for row in rows],
            xt=[
                seq_cat(seq_for_key[protein_keys[col]])
                for col in cols
            ],
            y=affinity[rows, cols],
            smile_graph=smile_graph,
            smile_tensor=smile_tensor,
            target_graph={
                key: target_graph[key][:3]
                for key in keys
            },
            target_key=keys,
            kmer_map=kmer_map,
            rnafm_map={
                key: target_graph[key][3]
                for key in keys
            },
            ligand_ids=ligand_ids[rows],
            drug_emb_map=drug_emb_map,
        )

    return (
        make_split("train", train_rows, train_cols),
        make_split("val", val_rows, val_cols),
        make_split("test", test_rows, test_cols),
    )
