# for lr in 8e-5 5e-5; do 
#     for sched in "constant" "inverse_sqrt"; do
#         for warmup in 1000 10000; do
#             for loss in "karras" "vanilla" "simple"; do
#                 slurm_run "${BASH_SOURCE[0]}" $lr $sched $warmup $loss
#             done
#         done
#     done
# done

lr=8e-5
warmup=10000
sched=inverse_sqrt
for loss in "karras" "vanilla" "simple"; do
    sbatch train.slrm $lr $sched $warmup $loss
done

lr=1e-5
warmup=1000
sched=constant
for loss in "karras" "vanilla" "simple"; do
    sbatch train.slrm $lr $sched $warmup $loss
done