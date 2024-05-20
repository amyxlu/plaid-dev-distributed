# for id in 1b64t79h 1hr1x9r5 yfel5fnl v2cer77t 2tjrgcde 3rs1hxky 5z4iaak9 q3m9fhii; do
#     sbatch run_benchmark.slrm --config-name $task ++task.model.compression_model_id=$id; sleep 10 
# done


task="human"
for id in 2tjrgcde 5z4iaak9 q3m9fhii; do
    sbatch run_benchmark.slrm --config-name $task ++task.model.compression_model_id=$id; sleep 10 
done


task="yeast"
for id in 2tjrgcde 5z4iaak9 q3m9fhii; do
    sbatch run_benchmark.slrm --config-name $task ++task.model.compression_model_id=$id; sleep 10 
done


task="contact"
for id in 3rs1hxky 5z4iaak9 q3m9fhii; do
    sbatch run_benchmark.slrm --config-name $task ++task.model.compression_model_id=$id; sleep 10 
done


task="ss"
for id in 1hr1x9r5 3rs1hxky 5z4iaak9 q3m9fhii; do
    sbatch run_benchmark.slrm --config-name $task ++task.model.compression_model_id=$id; sleep 10 
done