#### scaling #####
for model_id in 4hdab8dn 6ryvfi2v; do
    for length in 48 100 148; do
        sbatch scaling.slrm $model_id $length
    done
done


# ##### abalations #####
# sampdir=/data/lux70/plaid/artifacts/samples/ablations

# for model_id in lqp25b7g ksme77o6 5j007z42 q133mkem vzi7fsts ajaucr8g m91o13nd 9xd5bpc1 qfvl29in o2cgf3eq dojwljt5 jgxkivbq oi96ynxl k7nfljjc iu72i7c5 6ryvfi2v zlkurtdd 93qqcdh9 oa5kjy9x ye1j29dh 87up71bi f0luhi8y fo6tv2gf; do
#     for length in 100 148 48; do
#         sbatch scaling.slrm $model_id $length 
#     done
# done
