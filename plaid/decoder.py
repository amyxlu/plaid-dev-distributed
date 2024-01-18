import torch.nn as nn
import torch
import lightning as L

from typing import Optional, Callable

from plaid.losses import masked_token_cross_entropy_loss, masked_token_accuracy
from plaid.esmfold.misc import batch_encode_sequences
from plaid.transforms import get_random_sequence_crop_batch


class FullyConnectedNetwork(L.LightningModule):
    def __init__(
        self,
        n_classes: int,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 3,
        mlp_dropout_p: float = 0.1,
        add_sigmoid: bool = False,
        lr: float = 1e-4,
        esmfold: Optional[Callable] = None,
        training_max_seq_len: int = 512,
    ):
        super().__init__()
        self.batch_encode_sequences = batch_encode_sequences
        self.lr = lr
        if not esmfold is None:
            esmfold.requires_grad_(False).eval()
            esmfold.to(self.device)
        self.esmfold = esmfold
        self.training_max_seq_len = training_max_seq_len

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

            layers = (
                first_layer
                + second_layer
                + hidden_layer * num_hidden_layers
                + final_layer
            )

        if add_sigmoid:
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        self.save_hyperparameters()

    def forward(self, x):
        # for inference
        return self.net(x)
    
    def loss(self, logits, targets, mask=None):
        return masked_token_cross_entropy_loss(logits, targets, mask=mask) 
    
    def forward_pass_from_sequence(self, sequence):
        sequence = get_random_sequence_crop_batch(sequence, self.training_max_seq_len)
        aatype, mask, _, _, _ = self.batch_encode_sequences(sequence)
        latent = self.esmfold.infer_embedding(sequence)['s']
        aatype, mask = aatype.to(self.device), mask.to(self.device)
        return self(latent), aatype, mask

    def training_step(self, batch, batch_idx, **kwargs):
        _, sequence = batch
        logits, aatype, mask = self.forward_pass_from_sequence(sequence)
        loss = self.loss(logits, aatype, mask)
        acc = masked_token_accuracy(logits, aatype, mask=mask)
        self.log("train/loss", loss, batch_size=logits.shape[0], on_epoch=False, on_step=True)
        self.log("train/acc", acc, batch_size=logits.shape[0], on_epoch=False, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx, **kwargs):
        print("starting val step")
        _, sequence = batch
        logits, aatype, mask = self.forward_pass_from_sequence(sequence)
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
        decoder = cls()
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        decoder.load_state_dict(checkpoint["model_state_dict"])
        if eval_mode:
            for param in decoder.parameters():
                param.requires_grad = False
            decoder.eval()
        if not device is None:
            decoder.to(device)
        return decoder



if __name__ == "__main__":
    from plaid.datasets import FastaDataModule
    dm = FastaDataModule("/shared/amyxlu/data/uniref90/partial.fasta", batch_size=32)
    dm.setup("fit")
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))
    from plaid.esmfold import esmfold_v1
    esmfold = esmfold_v1()
    device = torch.device("cuda:6")
    esmfold = esmfold.to(device).eval().requires_grad_(False)
    module = FullyConnectedNetwork(n_classes=21, esmfold=esmfold, training_max_seq_len=512)
    module.to(device)
    import IPython; IPython.embed()
    module.training_step(batch, 0)