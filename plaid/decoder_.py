import torch.nn as nn
import numpy as np
import torch
import lightning as L

from typing import Optional, Callable
import wandb

from plaid.losses import masked_token_cross_entropy_loss, masked_token_accuracy
from plaid.esmfold.misc import batch_encode_sequences
from plaid.transforms import get_random_sequence_crop_batch


class FullyConnectedNetwork(torch.nn.Module):
    def __init__(
        self,
        n_classes: int = 21,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 3,
        mlp_dropout_p: float = 0.1,
        add_sigmoid: bool = False,
    ):
        self.n_classes = n_classes
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_layers = mlp_num_layers
        self.mlp_dropout_p = mlp_dropout_p
        self.add_sigmoid = add_sigmoid

        super().__init__()
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

    def forward(self, x):
        # for inference
        return self.net(x)


class Trainer:
    def __init__(
        self,
        model,
        device,
        train_dataloader,
        val_dataloader,
        ckpt_dir,
        esmfold,
        run_eval_every: int = 100,
        save_checkpoint_every: int = 100,
        log_every: int = 10,
        num_epochs: int = 100,
        lr=1e-4,
        training_max_seq_len=512,
        max_val_batches: Optional[int] = None,
    ):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.esmfold = esmfold
        self.run_eval_every = run_eval_every
        self.save_checkpoint_every = save_checkpoint_every
        self.log_every = log_every
        self.num_epochs = num_epochs
        self.checkpoint_path = ckpt_dir 
        self.lr = lr
        self.training_max_seq_len = training_max_seq_len
        self.max_val_batches = max_val_batches

        self.model = self.model.to(self.device)
        self.esmfold = self.esmfold.to(self.device).eval().requires_grad_(False)

        self.loss_fn = masked_token_cross_entropy_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.global_step = 0

    def loss(self, logits, targets, mask=None):
        return self.loss_fn(logits, targets, mask=mask)

    def forward_pass_from_sequence(self, sequence):
        sequence = get_random_sequence_crop_batch(sequence, self.training_max_seq_len)
        aatype, mask, _, _, _ = batch_encode_sequences(sequence)
        with torch.no_grad():
            latent = self.esmfold.infer(sequence)
        logits = self.model(latent)
        # also grab the aatype and mask labels
        aatype, mask = aatype.to(self.device), mask.to(self.device)
        del latent
        return logits, aatype, mask
    
    def run_batch(self, batch, train_mode=True):
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        _, sequence = batch
        logits, aatype, mask = self.forward_pass_from_sequence(sequence)

        loss = self.loss(logits, aatype, mask)
        acc = masked_token_accuracy(logits, aatype, mask=mask)
        prefix = "val"
        if train_mode:
            prefix = "train"
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        wandb.log({f"{prefix}/loss": loss})
        wandb.log({f"{prefix}/acc": acc})
        return loss, acc
    
    def run_eval(self):
        torch.cuda.empty_cache()
        all_loss = []
        all_acc = []
        for i, batch in enumerate(self.val_dataloader):
            if self.max_val_batches is not None and i >= self.max_val_batches:
                break 
            loss, acc = self.run_batch(batch, train_mode=False)
            all_loss.append(loss.item())
            all_acc.append(acc.item())
        wandb.log({"val/epoch_loss": np.mean(all_loss)})
        wandb.log({"val/epoch_acc": np.mean(all_acc)})
    
    def save_checkpoint(self):
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True)
        
        ckpt_path = self.ckpt_dir / f"checkpoint_{self.global_step}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr": self.lr,
                "n_classes": self.model.n_classes,
                "mlp_num_layers": self.model.mlp_num_layers,
                "mlp_hidden_dim": self.model.mlp_hidden_dim,
                "add_sigmoid": self.model.add_sigmoid,
                "global_step": self.global_step,
            },
            ckpt_path
        )
    
    def run(self):
        for _ in range(self.num_epochs):
            for train_batch in self.train_dataloader:
                self.run_batch(train_batch, train_mode=True)
                if self.global_step % self.run_eval_every == 0:
                    self.run_eval()
                if self.global_step % self.save_checkpoint_every == 0:
                    self.save_checkpoint()
                self.global_step += 1


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, default='/shared/amyxlu/plaid/checkpoints/sequence_decoder')
    p.add_argument("--fasta_file", type=str, default='/shared/amyxlu/data/uniref90/partial.fasta')
    p.add_argument("--n_classes", type=int, default=21)
    p.add_argument("--mlp_num_layers", type=int, default=3)
    p.add_argument("--mlp_hidden_dim", type=int, default=1024)
    p.add_argument("--mlp_dropout_p", type=float, default=0.1)
    p.add_argument("--add_sigmoid", action="store_true")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--training_max_seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--run_eval_every", type=int, default=100)
    p.add_argument("--save_checkpoint_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--max_val_batches", type=int, default=5)
    args = p.parse_args()
    return args

if __name__ == "__main__":
    from plaid.datasets import FastaDataModule
    from plaid.esmfold import esmfold_v1
    from datetime import datetime
    from pathlib import Path
    import argparse
    args = get_args()

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    ckpt_dir = Path(args.ckpt_dir) / timestamp
    print("Saving checkpoints to", ckpt_dir)

    dm = FastaDataModule(args.fasta_file, train_frac=0.99, batch_size=args.batch_size)
    dm.setup("fit")
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    device = torch.device("cuda")
    esmfold = esmfold_v1()
    model = FullyConnectedNetwork(args.n_classes, args.mlp_hidden_dim, args.mlp_num_layers, args.mlp_dropout_p, args.add_sigmoid)
    model.to(device)
    wandb.init(project="plaid-sequence-decoder", config=args)

    trainer = Trainer(
        model,
        device,
        train_dataloader,
        val_dataloader,
        ckpt_dir=ckpt_dir,
        esmfold=esmfold,
        run_eval_every=args.run_eval_every,
        save_checkpoint_every=args.save_checkpoint_every,
        log_every=args.log_every,
        num_epochs=args.num_epochs,
        lr=args.lr,
        training_max_seq_len=args.max_seq_len,
        max_val_batches=args.max_val_batches,
    )
    trainer.run()
