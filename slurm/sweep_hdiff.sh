# sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++datamodule.num_workers=16 
# sbatch train_diffusion.slrm experiment=hdif/pfam_clan ++datamodule.num_workers=16 ++callbacks.sample.batch_size=32 ++diffusion.lr=1e-5 
# sbatch train_diffusion.slrm experiment=hdif/pfam_clan ++resume_from_model_id='2dj696qw'
# sbatch train_diffusion.slrm experiment=hdif/pfam_clan ++resume_from_model_id='17lwgwcf'

for id in g8e83omk 7str7fhl ich20c3q uhg29zk4 13lltqha fbbrfqzk kyytc8i9 mm9fe6x9 8ebs7j9h; do
    sbatch train_diffusion.slrm experiment=hdif/cath_dit ++compression_model_id=$id
done