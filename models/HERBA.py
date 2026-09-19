import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    SAGPooling,
    global_mean_pool as gap,
    global_max_pool as gmp,
)
from torch_geometric.utils import to_dense_batch


class GatedBilinearFusion(nn.Module):
    def __init__(self, dim, rank=64, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.gate = nn.Linear(dim * 2, dim)
        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)
        self.Ux = nn.Linear(dim, rank, bias=False)
        self.Vy = nn.Linear(dim, rank, bias=False)
        self.fuse_proj = nn.Linear(rank, dim, bias=False)
        self.se_down = nn.Linear(
            dim, max(8, dim // 16)
        )
        self.se_up = nn.Linear(
            max(8, dim // 16), dim
        )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, y):
        gate = torch.sigmoid(
            self.gate(torch.cat([x, y], dim=-1))
        )
        hx = torch.tanh(self.proj_x(x))
        hy = torch.tanh(self.proj_y(y))
        gated = gate * hx + (1.0 - gate) * hy

        bilinear = self.fuse_proj(
            self.Ux(x) * self.Vy(y)
        )
        z = gated + bilinear

        scale = torch.relu(self.se_down(z))
        scale = torch.sigmoid(
            self.se_up(scale)
        )
        z = self.dropout(z * scale)

        z = self.ln(
            z + 0.1 * x + 0.1 * y
        )
        return self.ffn(z)


class NodeBidirectionalCrossAttention(nn.Module):


    def __init__(
        self,
        drug_dim,
        rna_dim,
        dim=128,
        heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.drug_proj = (
            nn.Identity()
            if drug_dim == dim
            else nn.Linear(drug_dim, dim)
        )
        self.rna_proj = (
            nn.Identity()
            if rna_dim == dim
            else nn.Linear(rna_dim, dim)
        )

        self.rna_queries_drug = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drug_queries_rna = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )

        self.rna_norm = nn.LayerNorm(dim)
        self.drug_norm = nn.LayerNorm(dim)


        self.gate = nn.Linear(
            dim * 2, dim
        )
        self.out = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
        )

    @staticmethod
    def masked_mean(x, mask):
        weight = mask.unsqueeze(-1).to(x.dtype)
        return (
            (x * weight).sum(dim=1)
            / weight.sum(dim=1).clamp_min(1.0)
        )

    def forward(
        self,
        drug_nodes,
        drug_batch,
        rna_nodes,
        rna_batch,
    ):
        drug, drug_mask = to_dense_batch(
            self.drug_proj(drug_nodes),
            drug_batch,
        )
        rna, rna_mask = to_dense_batch(
            self.rna_proj(rna_nodes),
            rna_batch,
        )

        rna_msg, _ = self.rna_queries_drug(
            rna,
            drug,
            drug,
            key_padding_mask=~drug_mask,
            need_weights=False,
        )
        drug_msg, _ = self.drug_queries_rna(
            drug,
            rna,
            rna,
            key_padding_mask=~rna_mask,
            need_weights=False,
        )

        rna_ctx = self.rna_norm(
            rna + rna_msg
        )
        drug_ctx = self.drug_norm(
            drug + drug_msg
        )

        rna_vec = self.masked_mean(
            rna_ctx, rna_mask
        )
        drug_vec = self.masked_mean(
            drug_ctx, drug_mask
        )

        gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [rna_vec, drug_vec],
                    dim=-1,
                )
            )
        )

        return self.out(
            torch.cat(
                [
                    gate * rna_vec,
                    (1.0 - gate) * drug_vec,
                ],
                dim=-1,
            )
        )


class ResidualMultiScaleNodeFusion(nn.Module):


    def __init__(self, d1, d2, d3, out_dim=128, gate_init=-2.0):
        super().__init__()
        self.proj1 = nn.Linear(d1, out_dim, bias=False)
        self.proj2 = nn.Linear(d2, out_dim, bias=False)
        self.proj3 = nn.Linear(d3, out_dim, bias=True)
        self.residual_gate = nn.Parameter(
            torch.full((out_dim,), float(gate_init))
        )

    def forward(self, x1, x2, x3):
        base = self.proj3(x3)
        earlier = 0.5 * (
            self.proj1(x1) + self.proj2(x2)
        )
        gate = torch.sigmoid(self.residual_gate).unsqueeze(0)
        return base + gate * earlier

class SeqTransformerEncoder(nn.Module):


    def __init__(
        self,
        embed_dim,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

    def forward(self, x):
        out = self.transformer(x)
        return out.mean(dim=1)


class MLSDTA(torch.nn.Module):
    def __init__(
        self,
        n_output=1,
        embed_dim=128,
        num_features_xd=78,
        num_features_xt=25,
        output_dim=128,
        dropout_rate=0.1,
        km_feature_dim=340,
        rnafm_dim=640,
    ):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        self.km_feature_dim = km_feature_dim
        self.rnafm_dim = rnafm_dim
        self.drug_emb_dim = 300


        self.drug_emb_mlp = nn.Sequential(
            nn.Linear(
                self.drug_emb_dim, 512
            ),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )
        self.bn_drug_emb = nn.LayerNorm(
            output_dim
        )

        self.gat_drug1 = GATConv(
            num_features_xd,
            num_features_xd,
            heads=4,
            concat=True,
        )
        self.gat_drug2 = GATConv(
            4 * num_features_xd,
            num_features_xd,
            heads=4,
            concat=True,
        )
        self.gat_drug3 = GATConv(
            4 * num_features_xd,
            num_features_xd,
            concat=False,
        )

        x1_out = num_features_xd * 4
        x2_out = num_features_xd * 4
        x3_out = num_features_xd
        linear_in_drug = (
            x1_out
            + x2_out
            + x3_out
            + self.drug_emb_dim
        )

        self.fc_druggraph = nn.Linear(
            linear_in_drug, output_dim
        )
        self.bn_fc_druggraph = nn.BatchNorm1d(
            output_dim
        )
        self.pool_drug = SAGPooling(
            linear_in_drug,
            ratio=0.5,
        )
        self.relu = nn.ReLU()


        self.num_features_xt = 54 + rnafm_dim

        self.gat_target1 = GATConv(
            self.num_features_xt,
            self.num_features_xt,
            heads=4,
            concat=True,
        )
        self.gat_target2 = GATConv(
            4 * self.num_features_xt,
            self.num_features_xt,
            heads=4,
            concat=True,
        )
        self.gat_target3 = GATConv(
            4 * self.num_features_xt,
            self.num_features_xt,
            concat=False,
        )

        xt1_out = self.num_features_xt * 4
        xt2_out = self.num_features_xt * 4
        xt3_out = self.num_features_xt
        linear_in_target = (
            xt1_out
            + xt2_out
            + xt3_out
            + rnafm_dim
        )

        self.fc_targetgraph = nn.Linear(
            linear_in_target, output_dim
        )
        self.bn_fc_targetgraph = nn.BatchNorm1d(
            output_dim
        )
        self.pool_target = SAGPooling(
            linear_in_target,
            ratio=0.5,
        )


        self.drug_node_fusion = ResidualMultiScaleNodeFusion(
            x1_out, x2_out, x3_out,
            out_dim=output_dim,
            gate_init=-2.0,
        )
        self.rna_node_fusion = ResidualMultiScaleNodeFusion(
            xt1_out, xt2_out, xt3_out,
            out_dim=output_dim,
            gate_init=-2.0,
        )



        self.embedding_xt = nn.Embedding(
            num_features_xt + 1,
            embed_dim,
        )
        self.seq_encoder_rna = SeqTransformerEncoder(
            embed_dim
        )
        self.embedding_xd = nn.Embedding(
            100,
            embed_dim,
        )
        self.seq_encoder_smiles = SeqTransformerEncoder(
            embed_dim
        )



        self.km_mlp = nn.Sequential(
            nn.Linear(km_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )
        self.bn_kmer = nn.LayerNorm(
            output_dim
        )

        self.rnafm_mlp = nn.Sequential(
            nn.Linear(rnafm_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )
        self.bn_rnafm = nn.LayerNorm(
            output_dim
        )


        dim = output_dim

        self.fuse_tx = GatedBilinearFusion(
            dim,
            rank=96,
            dropout=dropout_rate,
        )
        self.fuse_dx = GatedBilinearFusion(
            dim,
            rank=96,
            dropout=dropout_rate,
        )
        self.fuse_cross_g = (
            NodeBidirectionalCrossAttention(
                dim,
                dim,
                dim,
                heads=4,
                dropout=dropout_rate,
            )
        )

        self.fc_concat1 = nn.Linear(
            10 * output_dim,
            1024,
        )
        self.bn_fc_concat1 = nn.BatchNorm1d(
            1024
        )
        self.fc_concat2 = nn.Linear(
            1024,
            512,
        )
        self.bn_fc_concat2 = nn.BatchNorm1d(
            512
        )
        self.out = nn.Linear(
            512,
            n_output,
        )
        self.dropout = nn.Dropout(
            dropout_rate
        )

    def forward(
        self,
        DrugData,
        TargetData,
    ):

        x = DrugData.x
        edge_index = DrugData.edge_index
        batch = DrugData.batch

        x1 = self.relu(
            self.gat_drug1(
                x, edge_index
            )
        )
        x2 = self.relu(
            self.gat_drug2(
                x1, edge_index
            )
        )
        x3 = self.relu(
            self.gat_drug3(
                x2, edge_index
            )
        )


        drug_node_batch = batch
        drug_node_repr = self.drug_node_fusion(
            x1, x2, x3
        )

        graph_xd = torch.cat(
            [x1, x2, x3],
            dim=1,
        )
        graph_xd = torch.cat(
            [
                graph_xd,
                DrugData.drug_emb[
                    DrugData.batch
                ],
            ],
            dim=-1,
        )

        (
            graph_xd,
            edge_index,
            _,
            batch,
            *_
        ) = self.pool_drug(
            graph_xd,
            edge_index,
            batch=batch,
        )

        graph_xd = (
            gap(graph_xd, batch)
            + gmp(graph_xd, batch)
        ) / 2.0

        graph_xd = self.dropout(
            self.relu(
                self.bn_fc_druggraph(
                    self.fc_druggraph(
                        graph_xd
                    )
                )
            )
        )


        tx = TargetData.x
        target_edge_index = (
            TargetData.edge_index
        )
        tar_batch = TargetData.batch

        xt1 = self.relu(
            self.gat_target1(
                tx,
                target_edge_index,
            )
        )
        xt2 = self.relu(
            self.gat_target2(
                xt1,
                target_edge_index,
            )
        )
        xt3 = self.relu(
            self.gat_target3(
                xt2,
                target_edge_index,
            )
        )

        rna_node_batch = tar_batch
        rna_node_repr = self.rna_node_fusion(
            xt1, xt2, xt3
        )

        graph_xt = torch.cat(
            [xt1, xt2, xt3],
            dim=1,
        )
        graph_xt = torch.cat(
            [
                graph_xt,
                TargetData.rnafm[
                    TargetData.batch
                ],
            ],
            dim=-1,
        )

        (
            graph_xt,
            _,
            _,
            tar_batch,
            *_
        ) = self.pool_target(
            graph_xt,
            target_edge_index,
            batch=tar_batch,
        )

        graph_xt = (
            gap(graph_xt, tar_batch)
            + gmp(graph_xt, tar_batch)
        ) / 2.0

        graph_xt = self.dropout(
            self.relu(
                self.bn_fc_targetgraph(
                    self.fc_targetgraph(
                        graph_xt
                    )
                )
            )
        )



        seq_rna_feat = self.seq_encoder_rna(
            self.embedding_xt(
                TargetData.target
            )
        )
        seq_drug_feat = self.seq_encoder_smiles(
            self.embedding_xd(
                DrugData.smiles
            )
        )



        km = getattr(
            TargetData, "kmer", None
        )
        if km is not None:
            if km.dim() == 3:
                km = km.squeeze(1)
            km_feat = self.bn_kmer(
                self.km_mlp(km)
            )
        else:
            km_feat = torch.zeros(
                graph_xt.size(0),
                self.fc_targetgraph.out_features,
                device=graph_xt.device,
            )


        rf = getattr(
            TargetData, "rnafm", None
        )
        if rf is not None:
            if rf.dim() == 3:
                rf = rf.squeeze(1)
            rnafm_feat = self.bn_rnafm(
                self.rnafm_mlp(rf)
            )
        else:
            rnafm_feat = torch.zeros(
                graph_xt.size(0),
                self.fc_targetgraph.out_features,
                device=graph_xt.device,
            )


        emb = getattr(
            DrugData, "drug_emb", None
        )
        if emb is not None:
            if emb.dim() == 3:
                emb = emb.squeeze(1)
            drug_emb_feat = self.bn_drug_emb(
                self.drug_emb_mlp(emb)
            )
        else:
            drug_emb_feat = torch.zeros(
                graph_xd.size(0),
                self.fc_druggraph.out_features,
                device=graph_xd.device,
            )

        fused_target = self.fuse_tx(
            seq_rna_feat,
            graph_xt,
        )
        fused_drug = self.fuse_dx(
            seq_drug_feat,
            graph_xd,
        )

        fused_cross = self.fuse_cross_g(
            drug_node_repr,
            drug_node_batch,
            rna_node_repr,
            rna_node_batch,
        )

        concat_feat = torch.cat(
            [
                graph_xd,
                graph_xt,
                seq_drug_feat,
                seq_rna_feat,
                fused_target,
                fused_drug,
                fused_cross,
                km_feat,
                rnafm_feat,
                drug_emb_feat,
            ],
            dim=-1,
        )

        hidden = self.dropout(
            F.relu(
                self.bn_fc_concat1(
                    self.fc_concat1(
                        concat_feat
                    )
                )
            )
        )
        hidden = self.dropout(
            F.relu(
                self.bn_fc_concat2(
                    self.fc_concat2(
                        hidden
                    )
                )
            )
        )
        return self.out(hidden)
