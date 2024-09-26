# take 1:32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256
# take 2: 36 44 52 60 68 76 84 92 100 108 116 124 132 140 148 156 164 172 180 188 196 204 212 220 228 236 244 252

# arg_list=(32 40 48 56 64)
# sbatch evaluation_pipeline.slrm "${arg_list[@]}"

# arg_list=(72 80 88 96 104)
# sbatch evaluation_pipeline.slrm "${arg_list[@]}"

# arg_list=(112 120 128 136 144 152 160)
# sbatch evaluation_pipeline.slrm "${arg_list[@]}"

# arg_list=(168 176 184 192 200 208 216)
# sbatch evaluation_pipeline.slrm "${arg_list[@]}"

# arg_list=(224 232 240 248 256)
# sbatch evaluation_pipeline.slrm "${arg_list[@]}"


# ######

# arg_list=(36 44 52 60 68 76 84)
# sbatch evaluation_pipeline.slrm "${arg_list[@]}"

# arg_list=(92 100 108 116 124 132)
# sbatch evaluation_pipeline.slrm "${arg_list[@]}"

# arg_list=(140 148 156 164 172 180)
# sbatch evaluation_pipeline.slrm "${arg_list[@]}"

# arg_list=(188 196 204 212)
# sbatch evaluation_pipeline.slrm "${arg_list[@]}"


arg_list=(180 184 188 192 196 200)
sbatch evaluation_pipeline.slrm "${arg_list[@]}"

arg_list=(204 208 212 216 220)
sbatch evaluation_pipeline.slrm "${arg_list[@]}"

arg_list=(224 228 232 236 240)
sbatch evaluation_pipeline.slrm "${arg_list[@]}"

arg_list=(244 248 252 256)
sbatch evaluation_pipeline.slrm "${arg_list[@]}"