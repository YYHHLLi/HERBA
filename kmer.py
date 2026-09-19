import os
from itertools import product
import numpy as np

NUC = "ACGU"
NUC_SET = set(NUC)


def build_vocab(ks=(1, 2, 3, 4)):
    vocab = {}
    offset = 0
    for k in ks:
        for tup in product(NUC, repeat=k):
            vocab[(k, "".join(tup))] = offset
            offset += 1
    return vocab, offset


VOCAB, KM_DIM = build_vocab()


def _normalize_seq_rna(seq):
    seq = (seq or "").strip().upper().replace("T", "U")
    return "".join(ch for ch in seq if ch in NUC_SET)


def seq_to_kmer_freq(seq, ks=(1, 2, 3, 4), handle_ambig="skip"):
    if handle_ambig != "skip":
        raise ValueError("Only handle_ambig='skip' is supported.")
    s = _normalize_seq_rna(seq)
    vec = np.zeros(KM_DIM, dtype=np.float32)
    total = 0
    for k in ks:
        for i in range(max(0, len(s) - k + 1)):
            mer = s[i:i + k]
            if all(ch in NUC_SET for ch in mer):
                vec[VOCAB[(k, mer)]] += 1.0
                total += 1
    if total > 0:
        vec /= float(total)
    return vec


def fit_kmer_zscore(train_seqs, save_path, ks=(1, 2, 3, 4), handle_ambig="skip"):
    x = np.stack(
        [seq_to_kmer_freq(s, ks=ks, handle_ambig=handle_ambig) for s in train_seqs]
    )
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, mean=mean, std=std)
    return mean, std


def load_kmer_zscore(path):
    z = np.load(path)
    return z["mean"].astype(np.float32), z["std"].astype(np.float32)


def zscore(vec, mean, std):
    return (vec - mean) / std
