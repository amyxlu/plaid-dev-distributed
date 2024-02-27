import torch
import torch.nn as nn
import torch.nn.functional as F

# helpers
# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


# modified from https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/losses/contperceptual.py
# replace the GAN parts and perceptual loss (can add that back in outer loop)
class VAELoss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0, disc_factor=1.0, disc_weight=1.0,
                disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    # def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
    #     if last_layer is not None:
    #         nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    #         g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    #     else:
    #         nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
    #         g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

    #     d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    #     d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    #     d_weight = d_weight * self.discriminator_weight
    #     return d_weight

    def forward(self, inputs, reconstructions, posteriors, 
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = weighted_nll_loss + self.kl_weight * kl_loss 

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                }
        return loss, log
