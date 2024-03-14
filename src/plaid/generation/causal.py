from transformers import LlamaConfig, LlamaForCausalLM
import torch
from plaid.datasets import TokenDataModule
from plaid.utils import get_lr_scheduler

# """
# Set up
# """
# vocab_size = 128
# bos_token_id = vocab_size + 1
# eos_token_id = vocab_size + 2

# llama_config = LlamaConfig(
#     bos_token_id = bos_token_id,
#     eos_token_id = eos_token_id,
#     vocab_size = vocab_size,
#     num_hidden_layers = 4,
#     num_attention_heads = 8
# )

# print('loading llama...')
# model = LlamaForCausalLM(llama_config)

# device = torch.device('cuda')
# model = model.to(device)


"""
Training
"""
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# for i, batch in enumerate(train_dataloader):
#     if i > 3: break

#     batch = batch.long()  # chartensor -> long
#     batch = batch.to(device)
#     batch = batch[:, :64, :]
#     x = batch.reshape(batch.shape[0], -1)

#     """ Sample data batch
#     In [8]: x.shape
#     Out[8]: torch.Size([4, 1024])

#     In [9]: x
#     Out[9]:
#     tensor([[ 45,  89,  46,  ...,  97,  84,   4],
#             [106,  35,  35,  ...,  50,  74,  26],
#             [  0,  86,   6,  ...,  98,  74,  40],
#             [  1,  86,   0,  ...,  47,  79,   3]], device='cuda:0')
#     """
#     output = model(input_ids=x, labels=x)
#     pred_idx = output['logits'].argmax(-1)

#     loss = output['loss']

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(loss.item())

import lightning as L

class CausalLightningModule(L.LightningModule):
    def __init__(
        self,
        n_heads=8,
        n_layers=4,
        vocab_size=128,  # should match model used for data
    ):
        super().__init__()

        self.vocab_size = vocab_size
        hg_config = LlamaConfig(
            bos_token_id=vocab_size+1,
            eos_token_id=vocab_size+2,
            vocab_size=vocab_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads
        )

        self.model = LlamaForCausalLM(hg_config)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.enc.parameters()) + list(self.dec.parameters()) + list(self.quantizer.parameters()),
            lr=self.lr
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

if __name__ == "__main__":

    """
    Data
    """
    dm = TokenDataModule(batch_size=4)
    dm.setup()
    val_dataloader = dm.val_dataloader()
    train_dataloader = dm.train_dataloader()


    device = torch.device("cuda")
    model = CausalLightningModule(
        n_heads=8,
        n_layers=4,
        vocab_size=128
    )
    model.to(device)

    batch = next(iter(train_dataloader))
    batch = batch.to(device)
    import IPython;IPython.embed()
    model.training_step(batch, 0)
        
 

