#### scaling #####
# for model_id in 4hdab8dn 6ryvfi2v; do
#     for length in 48 100 148; do
#         sbatch scaling.slrm $model_id $length
#     done
# done

# for length in 48 100 148; do
#     sbatch scaling.slrm "6ryvfi2v" $length
# done


# for length in 48 100 148 200 248 300; do
#     for model_id in "btmop0c8" "reklt5kg"; do 
#         sbatch scaling.slrm $model_id $length
#     done
# done

# for length in 200 248 300; do
#     for model_id in "ksme77o6" "5j007z42" "6ryvfi2v"; do
#         sbatch scaling.slrm $model_id $length
#     done
# done
sampdir="/data/lux70/plaid/artifacts/samples/scaling"
for model_id in "ksme77o6" "5j007z42" "6ryvfi2v" "btmop0c8" "reklt5kg"; do
    for length in 248 300; do
        sbatch run_foldseek_only.slrm $sampdir/$model_id/$length 
        sbatch run_foldseek_only.slrm $sampdir/$model_id/$length --use_designability_filter
        sbatch run_mmseqs_only.slrm $sampdir/$model_id/$length  
    done
done

# ##### abalations #####
# sampdir=/data/lux70/plaid/artifacts/samples/ablations

# for model_id in lqp25b7g ksme77o6 5j007z42 q133mkem vzi7fsts ajaucr8g m91o13nd 9xd5bpc1 qfvl29in o2cgf3eq dojwljt5 jgxkivbq oi96ynxl k7nfljjc iu72i7c5 6ryvfi2v zlkurtdd 93qqcdh9 oa5kjy9x ye1j29dh 87up71bi f0luhi8y fo6tv2gf; do
#     for length in 100 148 48; do
#         sbatch scaling.slrm $model_id $length 
#     done
# done


