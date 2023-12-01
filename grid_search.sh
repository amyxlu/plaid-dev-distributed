lr=8e-5
warmup=10000
sched=inverse_sqrt
for loss in "karras" "simple"; do
    for density in "cosine" "cosine-interpolated"; do
        sbatch train.slrm $lr $sched $warmup $loss $density 
    done
done

lr=1e-5
warmup=1000
sched=constant
for loss in "karras" "simple"; do
    for density in "cosine" "cosine-interpolated" "lognormal"; do
        sbatch train.slrm $lr $sched $warmup $loss $density
    done
done