from math import sqrt
import numpy as np
import torch
from torch_geometric import data as DATA
from torch_geometric.data import Batch



class TestbedDataset(torch.utils.data.Dataset):
    """Paired HERBA dataset returning (DrugData, TargetData).

    Dataset caching is handled explicitly by train.py/test.py. Therefore this
    class intentionally inherits torch.utils.data.Dataset instead of PyG
    InMemoryDataset.
    """

    def __init__(
        self,
        root="/tmp",
        dataset="RNA",
        xd=None,
        xt=None,
        y=None,
        transform=None,
        pre_transform=None,
        smile_graph=None,
        smile_tensor=None,
        target_graph=None,
        target_key=None,
        kmer_map=None,
        rnafm_map=None,
        drug_emb_map=None,
        ligand_ids=None,
    ):
        self.root = root
        self.dataset = dataset
        self.transform = transform
        self.pre_transform = pre_transform
        self.DrugData = []
        self.TargetData = []

        xd = [] if xd is None else xd
        xt = [] if xt is None else xt
        y = [] if y is None else y
        smile_graph = {} if smile_graph is None else smile_graph
        smile_tensor = {} if smile_tensor is None else smile_tensor
        target_graph = {} if target_graph is None else target_graph
        target_key = [] if target_key is None else target_key

        self._build(
            xd, xt, y,
            smile_graph, smile_tensor,
            target_graph, target_key,
            kmer_map, rnafm_map,
            drug_emb_map, ligand_ids,
        )

    def _build(
        self,
        xd, xt, y,
        smile_graph, smile_tensor,
        target_graph, target_key,
        kmer_map=None,
        rnafm_map=None,
        drug_emb_map=None,
        ligand_ids=None,
    ):
        assert len(xd) == len(xt) == len(y), (
            f"length mismatch: xd={len(xd)}, xt={len(xt)}, y={len(y)}"
        )
        assert len(target_key) == len(y), (
            f"target_key={len(target_key)} but y={len(y)}"
        )
        if ligand_ids is not None:
            assert len(ligand_ids) == len(y), (
                f"ligand_ids={len(ligand_ids)} but y={len(y)}"
            )

        rnafm_dim = int(__import__("os").environ.get("RNAFM_DIM", 640))

        for i, (smiles, target_tokens, label, key) in enumerate(
            zip(xd, xt, y, target_key)
        ):
            c_size, features, edge_index = smile_graph[smiles]

            drug = DATA.Data(
                x=torch.tensor(np.asarray(features), dtype=torch.float32),
                edge_index=torch.tensor(
                    np.asarray(edge_index), dtype=torch.long
                ).t().contiguous(),
                y=torch.tensor([label], dtype=torch.float32),
            )
            drug.smiles = torch.from_numpy(
                np.asarray(smile_tensor[smiles], dtype=np.int64)
            ).unsqueeze(0)
            drug.c_size = torch.tensor([c_size], dtype=torch.long)

            emb = np.zeros(300, dtype=np.float32)
            if drug_emb_map is not None and ligand_ids is not None:
                mol_id = str(ligand_ids[i])
                if mol_id in drug_emb_map:
                    emb = np.asarray(drug_emb_map[mol_id], dtype=np.float32)
            drug.drug_emb = torch.from_numpy(emb).unsqueeze(0)

            tar_size, tar_features, tar_edge_index = target_graph[key]
            target = DATA.Data(
                x=torch.tensor(np.asarray(tar_features), dtype=torch.float32),
                edge_index=torch.tensor(
                    np.asarray(tar_edge_index), dtype=torch.long
                ).t().contiguous(),
                y=torch.tensor([label], dtype=torch.float32),
            )
            target.target = torch.from_numpy(
                np.asarray(target_tokens, dtype=np.int64)
            ).unsqueeze(0)
            target.tar_size = torch.tensor([tar_size], dtype=torch.long)

            km = np.zeros(340, dtype=np.float32)
            if kmer_map is not None and str(key) in kmer_map:
                km = np.asarray(kmer_map[str(key)], dtype=np.float32)
            target.kmer = torch.from_numpy(km).unsqueeze(0)

            rf = np.zeros(rnafm_dim, dtype=np.float32)
            if rnafm_map is not None and str(key) in rnafm_map:
                rf = np.asarray(rnafm_map[str(key)], dtype=np.float32)
            target.rnafm = torch.from_numpy(rf).unsqueeze(0)

            self.DrugData.append(drug)
            self.TargetData.append(target)

        print("data_len:", len(self.DrugData))

    def __len__(self):
        return len(self.DrugData)

    def __getitem__(self, idx):
        return self.DrugData[idx], self.TargetData[idx]


def collate(items):
    return (
        Batch.from_data_list([item[0] for item in items]),
        Batch.from_data_list([item[1] for item in items]),
    )


def train(model, device, train_loader, optimizer, epoch, ema=None):
    model.train()
    loss_fn = torch.nn.MSELoss()
    running = 0.0
    n_batch = 0

    for drug, target in train_loader:
        drug = drug.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(drug, target)
        labels = drug.y.view(-1, 1).float()

        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update(model)

        running += float(loss.item())
        n_batch += 1

    mean_loss = running / max(1, n_batch)
    print(f"epoch={epoch} train_mse={mean_loss:.6f}")
    return mean_loss


@torch.no_grad()
def predicting(model, device, loader):
    model.eval()
    labels_all = []
    preds_all = []

    for drug, target in loader:
        drug = drug.to(device)
        target = target.to(device)
        output = model(drug, target)

        labels_all.append(drug.y.view(-1, 1).detach().cpu())
        preds_all.append(output.detach().cpu())

    return (
        torch.cat(labels_all, dim=0).numpy().ravel(),
        torch.cat(preds_all, dim=0).numpy().ravel(),
    )


def rmse(y, f):
    return sqrt(float(np.mean((np.asarray(y) - np.asarray(f)) ** 2)))


def mse(y, f):
    return float(np.mean((np.asarray(y) - np.asarray(f)) ** 2))


def pearson(y, f):
    y = np.asarray(y)
    f = np.asarray(f)
    if len(y) < 2 or np.std(y) == 0 or np.std(f) == 0:
        return 0.0
    return float(np.corrcoef(y, f)[0, 1])


def _rankdata_average(x):
    x = np.asarray(x)
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(x))
    sx = x[sorter]

    starts = np.r_[0, np.nonzero(np.diff(sx))[0] + 1]
    ends = np.r_[starts[1:] - 1, len(sx) - 1]

    ranks = np.empty(len(sx), dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[start:end + 1] = (start + end) / 2.0 + 1.0
    return ranks[inv]


def spearman(y, f):
    return pearson(
        _rankdata_average(np.asarray(y)),
        _rankdata_average(np.asarray(f)),
    )


def ci(y, f):
    y = np.asarray(y)
    f = np.asarray(f)
    order = np.argsort(y)
    y = y[order]
    f = f[order]

    concordant = 0.0
    comparable = 0.0

    for i in range(1, len(y)):
        valid = y[i] > y[:i]
        delta = f[i] - f[:i]
        concordant += np.sum((delta > 0) & valid)
        concordant += 0.5 * np.sum((delta == 0) & valid)
        comparable += np.sum(valid)

    return float(concordant / comparable) if comparable > 0 else 0.0


def r_squared_error(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    y_bar = np.mean(y)
    p_bar = np.mean(p)
    num = np.sum((y - y_bar) * (p - p_bar)) ** 2
    den = np.sum((y - y_bar) ** 2) * np.sum((p - p_bar) ** 2)
    return float(num / den) if den > 0 else 0.0


def squared_error_zero(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    den_p = np.sum(p * p)
    k = np.sum(y * p) / den_p if den_p > 0 else 0.0
    den = np.sum((y - np.mean(y)) ** 2)
    if den <= 0:
        return 0.0
    return 1.0 - float(np.sum((y - k * p) ** 2) / den)


def rm2(y, p):
    r2 = r_squared_error(y, p)
    r02 = squared_error_zero(y, p)
    return float(r2 * (1.0 - np.sqrt(abs(r2 * r2 - r02 * r02))))
