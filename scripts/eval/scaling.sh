for model_id in 4hdab8dn 6ryvfi2v; do
    for length in 48 100 148; do
        sbatch scaling.slrm $model_id $length
    done
done