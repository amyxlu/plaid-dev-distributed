# import numpy as np
# import plaid as K
# import argparse
# import pandas as pd
# from pathlib import Path
# import torch

# parser = argparse.ArgumentParser()
# parser.add_argument("--model_name", type=str, default="bcjgpe29")
# parser.add_argument("--device_id", type=int, default=2)
# parser.add_argument("--step", type=int, default=3000)
# parser.add_argument("--num_recycles", type=int, default=4)
# parser.add_argument("--calc_perplexity", action="store_true")
# parser.add_argument("--batch_size", type=int, default=16)
# args = parser.parse_args()

# # import dataclasses
# # @dataclasses.dataclass
# # class Config:
# #     model_name: str = "bcjgpe29"
# #     device_id: int = 2
# #     step: int = 3000
# #     num_recycles = 4
# #     calc_perplexity = True
# #     batch_size = 16 
# # args = Config()


# device = torch.device(f"cuda:{args.device_id}")
# filepath = Path(
#     f"/home/amyxlu/kdiffusion/artifacts/sampled/{args.model_name}/step{args.step}_sampled.npy"
# )
# sampled_latent = np.load(
#     filepath,
#     allow_pickle=True,
# )
# latent = K.utils.to_tensor(sampled_latent, device=device)
# assert latent.shape[-1] == 1024

# sequence_constructor = K._proteins.LatentToSequence(device)
# structure_constructor = K._proteins.LatentToStructure(device)
# import IPython;IPython.embed()

# probs, _, sequences = sequence_constructor.to_sequence(latent)
# pdb_strs, metrics = structure_constructor.to_structure(
#     latent, sequences, args.num_recycles, batch_size=args.batch_size
# )

# sequence_results = pd.DataFrame(
#     {
#         "sequences": sequences,
#         "mean_residue_confidence": probs.mean(dim=1).cpu().numpy(),
#         # add additional log-to-disk metrics here
#     }
# )

# # maybe calculate sequence perplexities -- since RITA takes up memory, this is optional
# if args.calc_perplexity:
#     perplexity_calculator = K.evaluation.RITAPerplexity(device)
#     perplexities = perplexity_calculator.batch_calc_perplexity(sequences)
#     sequence_results["perplexity"] = K.utils.npy(perplexities)


# outdir = filepath.parent

# # write resulting sequences as a FASTA for downstream OmegaFold, etc.
# K.utils.write_to_fasta(outdir / "generated_sequences.fasta")

# # write auxiliary information to disk
# sequence_results.to_csv(outdir / "generated_sequences.csv", index=False)

# # write individual PDB strings to disk
# for i, pdbstr in enumerate(pdb_strs):
#     K.utils.write_pdb_to_disk(
#         pdbstr, outdir / "generated_structures" / f"sample_{i}.pdb"
#     )

#     # write auxiliary information to disk
#     metrics.to_csv(
#         outdir / "generated_structures" / "structure_confidence.csv", index=False
#     )
