from k_diffusion.models import ProteinTransformerDenoiserModelV1
import torch
model = ProteinTransformerDenoiserModelV1(
    n_layers=4,
    d_model=1024,
    d_ff=256,
    d_head=128,
    num_classes=0,
    dropout=0.0,
    sigma_data=1.0,
    input_size=512,
    min_len=32
)
device = torch.device("cuda:1")
model.to(device)

# simulate training setting where we have input sequences
# sequences = [
#     "DEYECDQESIARMLKLATHHQMVHNQFCMLWKVKHGGTPPGWKPFQWDNAKKHWDAEKLEGAWPSFPQQMIWIFKWTYEWS",
#     "SCTSWCIRFKPFEIIRDCFWISLSSMTTCNPRMFVKCHWFGRVKKLYQELHFTLKQPAVNVREIQDQKRYHAARWVYRWFCSTSPHMECRFALLIMQDHQ",
#     "VHWMPKYIAPFWHVQQEPQIKYGWRRGDFSIRSQPRCSLNCHNTEWNYDMGVMSVTPQCRFNWRENCENFCMKLFSNTNQVEATCWKTMDVESAVSWEDAESIARMLK",
#     "VLGHTGKMAWYDSIKHLQTEQSAAIDHAPSMGTEVLAFHQNMATVLNLSDRTINYQTYWNHPHPANFATIDVMDCFAPHAMTEANHRMCSGCHLNEQ",
#     "EKSAKDSFIGLHWITQQPSAPDQLPDQNGLSDHWGLRYEWGWQHFAVRMWDDYSFFAPGWTKTEFANGVMKRTDHSSN",
# ]



from torch.utils.data import random_split
from evo.dataset import FastaDataset
fasta_file = "/shared/amyxlu/data/uniref90/partial.fasta"

ds = FastaDataset(fasta_file, cache_indices=True)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

for i, batch in enumerate(loader):
    print("=========", "batch", i, "=========")
    sequences = batch[1]
    x, mask = model.embed_from_sequences(sequences)
    sigma = torch.rand(x.shape[0], 1)
    out = model(
        x=x.to(device),
        sigma=sigma.to(device),
        mask=mask.to(device),
        class_cond=None,
    )
    # import IPython; IPython.embed()

