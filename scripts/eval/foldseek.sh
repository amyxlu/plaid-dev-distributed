sampledir=/data/lux70/plaid/artifacts/samples/by_length/100
python foldseek.sh \
    -input_folder $sampledir \
    --next_folder /generated/structures/ \
    -outputfolder $sampledir/fold_seek_results_/ \
    -outputfile $sampledir/fold_seek_results \
    --f --d --n