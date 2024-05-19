# task=ss
# for id in 1b64t79h 1hr1x9r5 yfel5fnl v2cer77t 2tjrgcde 3rs1hxky 5z4iaak9 q3m9fhii; do
#     sbatch run_benchmark.slrm --config-name $task ++task.model.compression_model_id=$id; sleep 10 
# done

# id=q3m9fhii
# id=1b64t79h
# id=1hr1x9r5
id=yfel5fnl

for task in contact ss pdbbind bindingdb ppi_affinity yeast human; do
    sbatch run_benchmark.slrm --config-name $task ++task.model.compression_model_id=$id; sleep 10 
done
