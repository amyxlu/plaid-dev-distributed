
# function_idx=$1
# organism_idx=$2
# length=$3
# cond_scale=$4

function_idx=162

for length in 40 60 96; do
    for cond_scale in 1.0 2.0 4.0 8.0 16.0; do
        for organism_idx in 2436 1326 818 1452 300; do 
            echo $function_idx $organism_idx $length $cond_scale
            sbatch loop_conditional.slrm $function_idx $organism_idx $length $cond_scale 
        done
    done
done
# 758 ['BOVIN']
# 1411 ['CHICK']
# 2436 ['ECOLI']
# 158 ['HORSE']
# 1326 ['HUMAN']
# 1294 ['MAIZE']
# 300 ['MOUSE']
# 799 []
# 1265 ['RABIT']
# 716 []
# 333 ['SHEEP']
# 1357 ['SOYBN']
# 1388 ['TOBAC']
# 1452 ['WHEAT']
# 818 ['YEAST']