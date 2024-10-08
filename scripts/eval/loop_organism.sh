# for organism in 758 1411 2436 158 1326 1294 300 799 1265 716 333 1357 1388 1452 818; do
# for organism in 758 818 2436 1326; do
#     sbatch loop_organism.slrm $organism
# done

for ((org_idx=0; org_idx<=100; len+=2)); do
    sbatch loop_latent_only.slrm $org_idx 
done