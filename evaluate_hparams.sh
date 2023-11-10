for model_id in "6rpppdho" "39jxzyw2" "62cuu4lk" "vb0h6q5w"; do
    CUDA_VISIBLE_DEVICES=6 python sampling_callback.py --model-id $model_id --log-to-wandb --n-steps 15 --sequence-decode-strategy "onehot_categorical"
done 
