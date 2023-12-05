# lr=8e-5
# warmup=10000
# sched=inverse_sqrt
# for loss in "karras" "simple"; do
#     for density in "cosine" "cosine-interpolated"; do
#         sbatch train.slrm $lr $sched $warmup $loss $density 
#     done
# done

njobs=4
for i in $(seq 1 $njobs); do
    sbatch sweep.slrm
done