for model_id in "v2wureun" "3tbjkzg4" "ugn29pzs" "39jxzyw2" "62cuu4lk" "vb0h6q5w"; do 
    #CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 128 --n-to-construct 32 --model-id $model_id --log-to-wandb --n-steps 15 
    CUDA_VISIBLE_DEVICES=0 python sampling_callback.py --n-to-sample 32 --n-to-construct 32 --model-id $model_id  --n-steps 15 --solver-type "HEUN" --log-to-wandb 
done 

# model_id="6rpppdho"
# CUDA_VISIBLE_DEVICES=0 python -m pdb sampling_callback.py --n-to-sample 32 --n-to-construct 32 --model-id $model_id  --n-steps 15 --solver-type "DPMPP_2M_SDE"
    
