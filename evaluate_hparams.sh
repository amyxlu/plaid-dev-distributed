model_id="v2wureun"
model_step=50000
CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "HEUN" --log-to-wandb --calc-fid
CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPMPP_2M_SDE" --log-to-wandb --calc-fid
CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPM_2" --log-to-wandb --calc-fid
CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "LMS" --log-to-wandb --calc-fid
CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPM_2_ANCESTRAL" --log-to-wandb --calc-fid
CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPMPP_2S_ANCESTRAL" --log-to-wandb --calc-fid
CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPMPP_SDE" --log-to-wandb --calc-fid
CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPMPP_2M_SDE" --log-to-wandb --calc-fid
CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct -1 --model-id $model_id --model-step $model_step --n-steps 15 --solver-type "DPMPP_3M_SDE" --log-to-wandb --calc-fid

# model_id="6rpppdho"
# CUDA_VISIBLE_DEVICES=0 python -m pdb sampling_callback.py --n-to-sample 32 --n-to-construct 32 --model-id $model_id  --n-steps 15 --solver-type "DPMPP_2M_SDE"

