import os
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
)
import os as _os

_mof2zeo_dir = _os.path.dirname(_os.path.abspath(__file__))
__root_dir__ = _mof2zeo_dir


class MOFNET(pl.LightningModule):
    def __init__(self, config, scaler=None):

        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.lr = config["learning_rate"]
        self.scaler = scaler
        self.warmup_step = config["warmup_steps"]
        self.model = MOFEncoder(self.config)
        self.criterion = nn.MSELoss()
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y, x = batch
        y = y.squeeze()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, sync_dist=True, on_epoch=True, on_step=False)

        return loss

    def on_validation_start(self):
        self.validation_outputs = []

    def validation_step(self, batch, batch_idx):
        y, x = batch
        batch_size = x.shape[0]
        y = y.squeeze()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        output = {
            "val_loss": loss.item() * batch_size,
            "y_true": y,
            "y_pred": y_hat,
            "batch_size": batch_size,
        }

        self.validation_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_outputs
        total_samples = sum(x["batch_size"] for x in outputs)
        avg_loss = sum(x["val_loss"] for x in outputs) / total_samples
        y_true = torch.concat([x["y_true"] for x in outputs]).cpu().numpy()
        y_pred = torch.concat([x["y_pred"] for x in outputs]).detach().cpu().numpy()
        y_true_origin = self.scaler.decode(y_true)
        y_pred_origin = self.scaler.decode(y_pred)
        r2 = r2_score(y_true_origin, y_pred_origin)  # multioutput ='raw_values'
        mae = mean_absolute_error(y_true_origin, y_pred_origin)
        self.log(
            "val/avg_val_loss", avg_loss, sync_dist=True, on_epoch=True, on_step=False
        )
        self.log("val/avg_val_mae", mae, sync_dist=True, on_epoch=True, on_step=False)
        self.log(
            "val/avg_val_r2score", r2, sync_dist=True, on_epoch=True, on_step=False
        )

    def on_test_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        y, x = batch
        batch_size = x.shape[0]
        y = y.squeeze()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        output = {
            "test_loss": loss.item() * batch_size,
            "y_true": y,
            "y_pred": y_hat,
            "batch_size": batch_size,
        }

        self.test_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        outputs = self.test_outputs
        total_samples = sum(x["batch_size"] for x in outputs)
        avg_loss = sum(x["test_loss"] for x in outputs) / total_samples
        y_true = torch.concat([x["y_true"] for x in outputs]).cpu().numpy()
        y_pred = torch.concat([x["y_pred"] for x in outputs]).detach().cpu().numpy()
        y_true_origin = self.scaler.decode(y_true)
        y_pred_origin = self.scaler.decode(y_pred)
        r2 = r2_score(y_true_origin, y_pred_origin)  # multioutput ='raw_values'
        mae = mean_absolute_error(y_true_origin, y_pred_origin)
        self.log(
            "test/avg_test_loss", avg_loss, sync_dist=True, on_epoch=True, on_step=False
        )
        self.log("test/avg_test_mae", mae, sync_dist=True, on_epoch=True, on_step=False)
        self.log(
            "test/avg_test_r2score", r2, sync_dist=True, on_epoch=True, on_step=False
        )

        # save raw values
        feature_name_dir = self.config.get(
            "feature_name_dir", f"{__root_dir__}/data/feature_name.txt"
        )
        with open(feature_name_dir, "r") as g:
            feature_names = [line.strip() for line in g.readlines()]

        r2_raw = r2_score(y_true_origin, y_pred_origin, multioutput="raw_values")
        mae_raw = mean_absolute_error(
            y_true_origin, y_pred_origin, multioutput="raw_values"
        )
        results_df = pd.DataFrame(np.array([mae_raw, r2_raw]), columns=feature_names)
        results_df.insert(0, column="metric", value=["MAE", "r2"])
        results_df.to_csv(f"{self.config['exp_name']}_test_results.csv", index=None)

        # save values
        save_true = self.config.get("save_true", False)
        if save_true:
            np.save("test_origin.npy", y_true_origin)
            np.save("test_pred.npy", y_pred_origin)

    def on_predict_epoch_start(self):
        self.pred_outputs = []

    def predict_step(self, batch, batch_idx):
        token_ids = batch
        batch_size = token_ids.shape[0]

        y_hat = self(batch)

        self.pred_outputs.append(y_hat)

    def on_predict_epoch_end(self):
        all_pred = torch.concat(self.pred_outputs).detach().cpu().numpy()
        all_pred_origin = self.scaler.decode(all_pred)
        np.save(f"{self.config['exp_name']}_desc_pred.npy", all_pred_origin)

    def configure_optimizers(
        self,
    ):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        if self.trainer.max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
        else:
            max_steps = self.trainer.max_steps

        if isinstance(self.warmup_step, float):
            warmup_steps = int(max_steps * self.warmup_step)
        else:
            warmup_steps = self.warmup_step

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
        )

        sched = {"scheduler": scheduler, "interval": "step"}

        return (
            [optimizer],
            [sched],
        )


class MOFEncoder(nn.Module):
    """Mof encoder for a VAE."""

    def __init__(self, config):
        super().__init__()

        self.latent_dim = config.get("latent_dim", 128)
        self.hid_dim1 = config.get("hid_dim1", 64)
        self.hid_dim2 = config.get("hid_dim2", 32)
        self.desc_dim = config["desc_dim"]

        self.topo_emb = nn.Embedding(config["topo_size"], self.latent_dim)
        self.node_emb = nn.Embedding(config["node_size"], self.latent_dim)
        self.edge_emb = nn.Embedding(config["edge_size"], self.latent_dim)

        self.topo_emb_dropout = nn.Dropout()
        self.node_emb_dropout = nn.Dropout()
        self.edge_emb_dropout = nn.Dropout()

        self.self_node = nn.Linear(self.latent_dim, self.latent_dim * self.latent_dim)
        self.self_edge = nn.Linear(self.latent_dim, self.latent_dim * self.latent_dim)

        self.self_node_batchnorm = nn.BatchNorm2d(3)
        self.self_edge_batchnorm = nn.BatchNorm2d(3)

        self.self_node_tanh = nn.Tanh()
        self.self_edge_tanh = nn.Tanh()

        self.self_node_relu = nn.ReLU()
        self.self_edge_relu = nn.ReLU()

        self.inter = nn.Linear(self.latent_dim, 2 * self.latent_dim * self.hid_dim1)

        self.inter_batchnorm = nn.BatchNorm1d(self.hid_dim1)
        self.inter_relu = nn.ReLU()

        # Create a list of desc_dim small MLPs
        self.output_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hid_dim1, self.hid_dim2),
                    nn.BatchNorm1d(self.hid_dim2),
                    nn.ReLU(),
                    nn.Linear(self.hid_dim2, 1),
                )
                for _ in range(self.desc_dim)
            ]
        )

    def forward(self, mof_tensor):

        B = mof_tensor.shape[0]

        # (B,3) = topo(1) node(1) edge(1)
        topo_x, node_x, edge_x = torch.split(mof_tensor, [1, 1, 1], dim=1)

        topo_emb = self.topo_emb(topo_x)  # (B,1,E)
        node_emb = self.node_emb(node_x)  # (B,1,E)
        edge_emb = self.edge_emb(edge_x)  # (B,1,E)

        topo_emb = self.topo_emb_dropout(topo_emb)
        node_emb = self.node_emb_dropout(node_emb)
        edge_emb = self.edge_emb_dropout(edge_emb)

        # self-interaction weight: (B,1,E,E)
        node_weight = self.self_node(topo_emb)  # (B,1,E*E)
        edge_weight = self.self_edge(topo_emb)  # (B,1,E*E)
        node_weight = self.self_node_tanh(node_weight)
        edge_weight = self.self_edge_tanh(edge_weight)

        node_weight = node_weight.view(B, 1, self.latent_dim, self.latent_dim)
        edge_weight = edge_weight.view(B, 1, self.latent_dim, self.latent_dim)

        # (B,1,E,E) x (B,1,E) -> (B,1,E)
        node_emb = torch.einsum("b n i j, b n j -> b n i", node_weight, node_emb)
        edge_emb = torch.einsum("b n i j, b n j -> b n i", edge_weight, edge_emb)

        node_emb = self.self_node_relu(node_emb)
        edge_emb = self.self_edge_relu(edge_emb)

        # (B,1,E) -> (B,E)
        node_emb = node_emb.squeeze(1)
        edge_emb = edge_emb.squeeze(1)

        # concat: (B, 2E)
        x = torch.concat([node_emb, edge_emb], dim=1)

        # inter: (B,1,E) -> Linear -> (B,1,2E*hid1) -> (B,2E,hid1)
        inter_weight = self.inter(topo_emb)  # (B,1,2E*hid1)
        inter_weight = inter_weight.view(B, 2 * self.latent_dim, self.hid_dim1)

        # (B,2E,hid1) x (B,2E) -> (B,hid1)
        x = torch.einsum("b i o, b i -> b o", inter_weight, x)

        x = self.inter_batchnorm(x)
        x = self.inter_relu(x)

        outputs = [head(x) for head in self.output_heads]
        y = torch.cat(outputs, dim=1)
        return y
