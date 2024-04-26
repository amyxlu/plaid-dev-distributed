sbatch train_diffusion.slrm experiment=hdif/pfam_dit 
sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++diffusion.objective="pred_v"

sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++diffusion.x_downscale_factor=0.5
sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++diffusion.x_downscale_factor=0.3
sbatch train_diffusion.slrm experiment=hdif/pfam_dit ++diffusion.x_downscale_factor=0.1

sbatch train_diffusion.slrm experiment=hdif/pfam_dit_sigmoid
