#! /bin/bash

# sampdir=/data/lux70/plaid/artifacts/samples/scaling


# Base directory
BASE_DIR="/data/lux70/plaid/artifacts/samples/5j007z42/val_dist"

# Find all subdirectories
find "$BASE_DIR" -type d | while read -r dir; do
    # Check if generated/structures exists in this directory
    if [ -d "$dir/generated/structures" ]; then
        # Check if designability.csv does NOT exist
        if [ ! -f "$dir/designability.csv" ]; then
            echo "Found matching directory: $dir"
            
            # Add your action here
            # For example:
            # cd "$dir" && your_command_here
            
            # Uncomment and modify the line above to execute your desired action
        fi
    fi
done
# for ((len=32; len<=256; len+=4)); do
#     sbatch run_analysis.slrm $sampdir/$len
# done

# for subdir in "$sampdir"/*/; do
#   if [ -d "$subdir" ]; then
#     for len in 100 148 48; do
#         echo $subdir$len
#         sbatch run_analysis.slrm $subdir$len
#     done
#   fi
# done


# for * in sampdir; do
# for len in 100 148 48; do
#     for model_id in 6ryvfi2v 4hdab8dn; do
#         sbatch run_analysis.slrm $sampdir/$model_id/$len
#         sbatch run_analysis.slrm $sampdir/$model_id/$len
#     done
# done