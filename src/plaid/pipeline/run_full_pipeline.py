import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# from plaid.pipelines import ...

@hydra.main(version_base=None, config_path="configs", config_name="train_hourglass_vq")
def run_pipeline(cfg: DictConfig):
    if cfg.sample_latent:
        pass

    if cfg.calculate_fid:
        pass

    if cfg.generate_latent_to_sequence:
        pass

    if cfg.generate_latent_to_structure:
        pass

    if cfg.log_perplexity_and_plddt:
        pass

    if cfg.inverse_reconstruct_generated_sequence_to_reconstructed_structure:
        pass

    if cfg.inverse_reconstruct_generated_structure_to_reconstructed_sequence:
        pass

    if cfg.calculate_ccsr_and_ccrmsd:
        pass

    if cfg.phantom_reconstruct_generated_sequence_to_phantom_structure:
        pass

    if cfg.phantom_reconstruct_generated_structure_to_phantom_sequence:
        pass

    if cfg.calculate_scsr_and_scrmsd:
        pass


if __name__ == "__main__":
    run_pipeline()