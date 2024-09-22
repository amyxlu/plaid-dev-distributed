import torch.nn as nn
import torch
import lightning as L
import typing as T
from pathlib import Path
import os

from typing import Optional, Callable

from torch.nn import CrossEntropyLoss


class FullyConnectedNetwork(L.LightningModule):
    def __init__(
        self,
        n_classes: int,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 3,
        mlp_dropout_p: float = 0.1,
        add_sigmoid: bool = False,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.lr = lr

        if mlp_num_layers == 1:
            layers = [nn.Linear(mlp_hidden_dim, n_classes)]

        elif mlp_num_layers == 2:
            first_layer = [
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]
            final_layer = [
                nn.Linear(mlp_hidden_dim // 4, n_classes),
            ]
            layers = first_layer + final_layer

        else:
            assert mlp_num_layers >= 3
            num_hidden_layers = mlp_num_layers - 3

            first_layer = [
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]

            second_layer = [
                nn.Linear(mlp_hidden_dim // 2, mlp_hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]

            hidden_layer = [
                nn.Linear(mlp_hidden_dim // 4, mlp_hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]

            final_layer = [
                nn.Linear(mlp_hidden_dim // 4, n_classes),
            ]

            layers = first_layer + second_layer + hidden_layer * num_hidden_layers + final_layer

        if add_sigmoid:
            layers.append(nn.Sigmoid())

        self.criterion = CrossEntropyLoss()
        self.net = nn.Sequential(*layers)
        self.save_hyperparameters()

    def forward(self, x):
        # for inference
        return self.net(x)

    def loss(self, logits, targets, mask=None):
        return self.criterion(logits, targets)

    def forward_pass_from_sequence(self, sequence):
        latent = self.training_embed_from_sequence_fn(sequence)
        if self.latent_scaler is not None:
            latent = self.latent_scaler.scale(latent)
        return self(latent)

    def training_step(self, batch, batch_idx, **kwargs):
        _, sequence = batch
        sequence = get_random_sequence_crop_batch(sequence, self.training_max_seq_len)
        aatype, mask, _, _, _ = self.batch_encode_sequences(sequence)
        aatype, mask = aatype.to(self.device), mask.to(self.device)

        logits = self.forward_pass_from_sequence(sequence)

        loss = self.loss(logits, aatype, mask)
        acc = masked_token_accuracy(logits, aatype, mask=mask)
        self.log("train/loss", loss, batch_size=logits.shape[0], on_epoch=False, on_step=True)
        self.log("train/acc", acc, batch_size=logits.shape[0], on_epoch=False, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        print("starting val step")
        _, sequence = batch
        sequence = get_random_sequence_crop_batch(sequence, self.training_max_seq_len)
        aatype, mask, _, _, _ = self.batch_encode_sequences(sequence)
        aatype, mask = aatype.to(self.device), mask.to(self.device)

        logits = self.forward_pass_from_sequence(sequence)

        loss = self.loss(logits, aatype, mask)
        acc = masked_token_accuracy(logits, aatype, mask=mask)
        print("val loss: ", loss.item(), "val acc: ", acc.item())
        self.log("val/loss", loss, batch_size=logits.shape[0], on_epoch=False, on_step=True)
        self.log("val/acc", acc, batch_size=logits.shape[0], on_epoch=False, on_step=True)
        return loss

    def test_step(self, batch, batch_idx, **kwargs):
        pass
        # _, sequence = batch
        # logits, aatype, mask = self.forward_pass_from_sequence(sequence)
        # loss = self.loss(logits, aatype, mask)
        # acc = masked_token_accuracy(logits, aatype, mask=mask)
        # self.log("test/loss", loss, batch_size=logits.shape[0], on_epoch=False, on_step=True)
        # self.log("test/acc", acc, batch_size=logits.shape[0], on_epoch=False, on_step=True)
        # return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @classmethod
    def from_pretrained(cls, device=None, ckpt_path=None, eval_mode=True):
        if ckpt_path is None:
            ckpt_path = DECODER_CKPT_PATH
        model = cls.load_from_checkpoint(ckpt_path)
        if device is not None:
            model.to(device)
        if eval_mode:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        return model

