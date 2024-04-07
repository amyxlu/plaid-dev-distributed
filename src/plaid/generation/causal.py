from transformers import LlamaConfig, LlamaForCausalLM
import torch
from plaid.datasets import TokenDataModule
from plaid.utils import get_lr_scheduler

import lightning as L


class CausalLightningModule(L.LightningModule):
    def __init__(
        self,
        # llama config
        vocab_size=128,  # should match model used for data
        hidden_size: int = 4096,
        intermediate_size: int = 11108,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        max_position_embeddings: int = 2048,
        # lr:
        lr=1e-4,
        lr_adam_betas=(0.9, 0.999),
        lr_sched_type: str = "constant",
        lr_num_warmup_steps: int = 0,
        lr_num_training_steps: int = 10_000_000,
        lr_num_cycles: int = 1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        hg_config = LlamaConfig(
            bos_token_id=vocab_size+1,
            eos_token_id=vocab_size+2,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings
        )

        self.model = LlamaForCausalLM(hg_config)
        
        self.lr = lr
        self.lr_adam_betas = lr_adam_betas
        self.lr_sched_type = lr_sched_type
        self.lr_num_warmup_steps = lr_num_warmup_steps
        self.lr_num_training_steps = lr_num_training_steps
        self.lr_num_cycles = lr_num_cycles

        # TODO: maybe also add post-decoding sequence / structure losses?

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=self.lr_adam_betas
        )
        scheduler = get_lr_scheduler(
            optimizer=optimizer,
            sched_type=self.lr_sched_type,
            num_warmup_steps=self.lr_num_warmup_steps,
            num_training_steps=self.lr_num_training_steps,
            num_cycles=self.lr_num_cycles,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def forward(self, x, mask=None):
        # TODO: mask
        return self.model(input_ids=x, labels=x)
    
    def training_step(self, batch, batch_idx):
        # TODO: mask
        x = batch
        output = self(x)
        return output['loss']
    
    def validation_step(self, batch, batch_idx):
        # TODO: mask
        x = batch
        output = self(x)
        return output['loss']