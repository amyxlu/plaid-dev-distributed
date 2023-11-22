# model_id="v2wureun"
# model_step=50000
# CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "HEUN" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPMPP_2M_SDE" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPM_2" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "LMS" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPM_2_ANCESTRAL" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPMPP_2S_ANCESTRAL" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPMPP_SDE" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPMPP_2M_SDE" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPMPP_3M_SDE" --log-to-wandb --calc-fid

# model_id="6rpppdho"
# CUDA_VISIBLE_DEVICES=0 python -m pdb sampling_callback.py --n-to-sample 32 --n-to-construct 32 --model-id $model_id  --n-steps 15 --solver-type "DPMPP_2M_SDE"


# htkunb2s == j48t4tgo
# q4ww6po8 == rapsh8qe

device_id=$1


# model_id="rapsh8qe"
# model_step=45000
# CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "HEUN" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPM_2" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "LMS" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPM_2_ANCESTRAL" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPMPP_2S_ANCESTRAL" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPMPP_SDE" --log-to-wandb --calc-fid
# 
# model_id="vw6bi6r6"
# model_step=100000
# CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "HEUN" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPM_2" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "LMS" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPM_2_ANCESTRAL" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPMPP_2S_ANCESTRAL" --log-to-wandb --calc-fid
# CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPMPP_SDE" --log-to-wandb --calc-fid
# 
for model_id in "htkunb2s" "dnytqro5" "46ogp49p"; do
    CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --n-steps 15 --solver-type "HEUN" --log-to-wandb --calc-fid
    CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --n-steps 15 --solver-type "DPM_2" --log-to-wandb --calc-fid
    CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --n-steps 15 --solver-type "LMS" --log-to-wandb --calc-fid
    CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --n-steps 15 --solver-type "DPM_2_ANCESTRAL" --log-to-wandb --calc-fid
    CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --n-steps 15 --solver-type "DPMPP_2S_ANCESTRAL" --log-to-wandb --calc-fid
    CUDA_VISIBLE_DEVICES=$device_id python sampling_callback.py --n-to-sample 64 --n-to-construct 32 --model-id $model_id --n-steps 15 --solver-type "DPMPP_SDE" --log-to-wandb --calc-fid
done


