
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


# organism_idx=2436
# cond_scale=3
# length="None"
# subdir=""

# for function_idx in 1636 1666 820 880; do
#     echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale SubDir $subdir
#     sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir
# done

# # prokaryotic only functions:
# organism_idx=2436
# function_idx=46
# sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir


# length=212
# function_idx=323
# for organism_idx in 1326 300; do
#     sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir
# done

function_idx=356
cond_scale=2
length=112
subdir=""

for organism_idx in 818 2436 1326 3702 300; do
    echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale SubDir $subdir
    sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir
done

###################
# cond sclaing ablation 
###################

# organism_idx=1326
# length=256
# subdir="cond_scale"
# function_idx=5

# for cond_scale in 0 1 2 3 4 5 6 7 8 9 10; do
#     echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale SubDir $subdir
#     sbatch loop_compositional.slrm $function_idx $organism_idx $length $cond_scale $subdir
# done

