# for id in e2iwuio0 t1ji0n45 0u6sw28c 5006fhbi 6nahhz22 ftqrhze9 xppmuppx gzkug3bi; do
#     sbatch train_diffusion.slrm resume_from_model_id=$id
# done

# sbatch train_diffusion_2gpu.slrm experiment=hdif/pfam_dit datamodule.h5_root_dir=/homefs/home/lux70/storage/data/pfam/compressed/subset_30K_with_clan ++datamodule.batch_size=128 ++datamodule.aprclan_version=True
# python run_diffusion_slrm.py --n_gpus 4 --flags "experiment=hdif/cached_pfam_ddp ++datamodule.h5_root_dir=/homefs/home/lux70/storage/data/pfam/compressed/subset_30K_with_clan ++datamodule.batch_size=256" # ++datamodule.aprclan_version=True"
python run_diffusion_slrm.py --n_gpus 1 --flags "experiment=hdif/pfam_clan ++diffusion.sequence_decoder_weight=1.0"
