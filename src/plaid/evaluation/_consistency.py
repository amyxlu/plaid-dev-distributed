


        # # calculate the TM-scores with implicit alignment
        # tm_scores_list = self._run_tmalign(orig_pdb_paths, recons_pdb_paths)
        # n = len(tm_scores_list)
        # log_dict["tmscore_mean"] = np.mean(tm_scores_list)
        # log_dict["tmscore_median"] = np.median(tm_scores_list)
        # log_dict["tmscore_hist"] = wandb.Histogram(tm_scores_list, num_bins=n)

        # # pdb parse returns an atom stack; take only the first atom array
        # orig_atom_arrays = [pdb_path_to_biotite_atom_array(fpath)[0] for fpath in orig_pdb_paths]
        # recons_atom_arrays = [pdb_path_to_biotite_atom_array(fpath)[0] for fpath in recons_pdb_paths]
        # recons_superimposed = [
        #     structure.superimpose(orig, recons)[0]
        #     for (orig, recons) in zip(orig_atom_arrays, recons_atom_arrays)
        # ]

        # # calculate superimposed RMSD
        # superimposed_rmsd_scores = [
        #     structure.rmsd(orig, recons) for (orig, recons) in zip(orig_atom_arrays, recons_superimposed)
        # ]
        # log_dict["rmsd_mean"] = np.mean(superimposed_rmsd_scores)
        # log_dict["rmsd_median"] = np.median(superimposed_rmsd_scores)
        # log_dict["rmsd_hist"] = wandb.Histogram(superimposed_rmsd_scores, num_bins=n)

        # # calculate lDDT from alpha carbons
        # orig_ca_pos = [alpha_carbons_from_atom_array(aarr) for aarr in orig_atom_arrays]
        # recons_ca_pos = [alpha_carbons_from_atom_array(aarr) for aarr in recons_superimposed]
        # lddts = [
        #     lDDT(
        #         to_tensor(orig_ca_pos[i].coord),
        #         to_tensor(recons_ca_pos[i].coord),
        #     )
        #     for i in range(len(orig_ca_pos))
        # ]
        # log_dict["lddt_mean"] = np.mean(lddts)
        # log_dict["lddt_median"] = np.median(lddts)
        # log_dict["lddt_hist"] = wandb.Histogram(lddts, num_bins=n)

        # # calculate RMSD between pairwise distances (superimposition independent)
        # rmspd_scores = [
        #     structure.rmspd(orig, recons) for (orig, recons) in zip(orig_atom_arrays, recons_superimposed)
        # ]
        # log_dict["rmspd_mean"] = np.mean(rmspd_scores)
        # log_dict["rmspd_median"] = np.median(rmspd_scores)
        # log_dict["rmspd_hist"] = wandb.Histogram(rmspd_scores, num_bins=n)

        # end = time.time()
        # print(f"Structure reconstruction validation completed in {end - start:.2f} seconds.")
        # return log_dict
