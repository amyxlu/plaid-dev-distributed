
# function_idx=$1
# organism_idx=$2
# length=$3
# cond_scale=$4

### DNA transcription initiator
# function_idx=162
# for length in 40 60 96; do

### Protein deubiquitinase
# for length in 40 88 100 160 196 212; do
#     for cond_scale in 2 4 8; do
#         for organism_idx in 2436 1326 818 1452 300; do 
#             echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale
#             sbatch loop_conditional.slrm $function_idx $organism_idx $length $cond_scale 
#         done
#     done
# done

### Protein kinase
# function_idx=169
# for length in 10 20 40 80 120 160; do
#     for cond_scale in 3; do
#         for organism_idx in 2436 1326 818 1452 300; do 
#             echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale
#             sbatch loop_conditional.slrm $function_idx $organism_idx $length $cond_scale 
#         done
#     done
# done


### membrane
# function_idx=64
# for length in 20 40 80 120 160; do
#     for cond_scale in 3; do
#         for organism_idx in 2436 1326 818 1452 300; do 
#             echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale
#             sbatch loop_conditional.slrm $function_idx $organism_idx $length $cond_scale 
#         done
#     done
# done


### ribosome
# function_idx=74
# for length in 20 40 80 120 160; do
#     for cond_scale in 3; do
#         for organism_idx in 2436 1326 818 1452 300; do 
#             echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale
#             sbatch loop_conditional.slrm $function_idx $organism_idx $length $cond_scale 
#         done
#     done
# done


## metal ion binding
# function_idx=38
# for length in 10 20 40 80; do
#     for cond_scale in 3; do
#         for organism_idx in 2436 1326 818 1452 300; do 
#             echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale
#             sbatch loop_conditional.slrm $function_idx $organism_idx $length $cond_scale 
#         done
#     done
# done

# protein kinase activity
# function_idx=28
# for length in 116 200; do
#     for cond_scale in 3 8; do
#         for organism_idx in 1326 300; do 
#             echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale
#             sbatch loop_conditional.slrm $function_idx $organism_idx $length $cond_scale 
#         done
#     done
# done

# organism_idx=1326
# function_idx=2219
# for length in 116 200; do
#     for cond_scale in 3 8; do
#         echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale
#         sbatch loop_conditional.slrm $function_idx $organism_idx $length $cond_scale 
#     done
# done

# organism_idx=300
# function_idx=2219
# for length in 116 200; do
#     for cond_scale in 3 8; do
#         echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale
#         sbatch loop_conditional.slrm $function_idx $organism_idx $length $cond_scale 
#     done
# done

# organism_idx=3617
# function_idx=28
# for length in 116 200; do
#     for cond_scale in 3 8; do
#         echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale
#         sbatch loop_conditional.slrm $function_idx $organism_idx $length $cond_scale 
#     done
# done


# function_idx=475
# for length in 32 94 128; do
#     for cond_scale in 3 7; do
#         for organism_idx in 2436 1326 818 1452 300; do 
#             echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale
#             sbatch loop_conditional.slrm $function_idx $organism_idx $length $cond_scale 
#         done
#     done
# done


# function_idx=727
# for length in 32 94 128; do
#     for cond_scale in 3 7; do
#         for organism_idx in 2436 1326 818 1452 300; do 
#             echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale
#             sbatch loop_conditional.slrm $function_idx $organism_idx $length $cond_scale 
#         done
#     done
# done
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


# organism_idx=3617
# function_idx=166
# cond_scale=2
# length=1000
# subdir=""

# for organism_idx in 1326 300 1398 818 2436 3617; do
#     echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale SubDir $subdir
#     sbatch loop_conditional.slrm $function_idx $organism_idx $length $cond_scale $subdir
# done

organism_idx=3617
cond_scale=3
length="None"
subdir=""

for function_idx in {1..100}; do
    echo Function $function_idx Organism $organism_idx Length $length CondScale $cond_scale SubDir $subdir
    # sbatch loop_conditional.slrm $function_idx $organism_idx $length $cond_scale $subdir
done