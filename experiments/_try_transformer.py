from k_diffusion.models.protein_transformer_v1 import  TransformerDenoiserModelV1
import torch
model = TransformerDenoiserModelV1(
    n_layers=4,
    d_model=1024,
    d_ff=256,
    d_head=128,
    num_classes=0,
    dropout=0.0,
    sigma_data=1.0,
)
device = torch.device("cuda:1")
model.to(device)

# simulate training setting where we have input sequences
sequences = [
    "DEYECDQESIARMLKLATHHQMVHNQFCMLWKVKHGGTPPGWKPFQWDNAKKHWDAEKLEGAWPSFPQQMIWIFKWTYEWS",
    "SCTSWCIRFKPFEIIRDCFWISLSSMTTCNPRMFVKCHWFGRVKKLYQELHFTLKQPAVNVREIQDQKRYHAARWVYRWFCSTSPHMECRFALLIMQDHQ",
    "VHWMPKYIAPFWHVQQEPQIKYGWRRGDFSIRSQPRCSLNCHNTEWNYDMGVMSVTPQCRFNWRENCENFCMKLFSNTNQVEATCWKTMDVESAVSWEDAESIARMLK",
    "VLGHTGKMAWYDSIKHLQTEQSAAIDHAPSMGTEVLAFHQNMATVLNLSDRTINYQTYWNHPHPANFATIDVMDCFAPHAMTEANHRMCSGCHLNEQ",
    "EKSAKDSFIGLHWITQQPSAPDQLPDQNGLSDHWGLRYEWGWQHFAVRMWDDYSFFAPGWTKTEFANGVMKRTDHSSN",
]

x, mask = model.embed_from_sequences(sequences)
sigma = torch.rand(x.shape[0], 1)
out = model(
    x=x.to(device),
    mask=mask.to(device),
    sigma=sigma.to(device),
    aug_cond=None,
    class_cond=None,
)
# import IPython; IPython.embed()

