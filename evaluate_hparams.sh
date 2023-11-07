for model_id in "7t63xkvs" "7m86038f" "tqv1s5ce" "yqwdco11"; do
    CUDA_VISIBLE_DEVICES=6 python sampling_callback.py --model-id $model_id
done 
