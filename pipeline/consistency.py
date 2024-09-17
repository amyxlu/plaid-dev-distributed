from pathlib import Path
import glob

from plaid.evaluation import batch_rmsd_calculation, batch_rmspd_from_pdb_paths, run_tmalign
from plaid.utils import read_sequences_from_fasta, calc_sequence_recovery


class CrossConsistencyEvaluation:
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)

        generated_pdb_dir = self.experiment_dir / "generated" / "structures"
        self.generated_pdb_paths = glob.glob(str(generated_pdb_dir / "*.pdb"))
        self.generated_pdb_paths.sort()

        self.generated_fasta = self.experiment_dir / "generated" / "sequences.fasta"

        inverse_generated_pdb_dir = self.experiment_dir / "inverse_generated" / "structures"
        self.inverse_generated_pdb_paths = glob.glob(str(inverse_generated_pdb_dir / "*.pdb"))
        self.inverse_generated_pdb_paths.sort()

        self.inverse_generated_fasta = self.experiment_dir / "inverse_generated" / "sequences.fasta"

    def cross_consistency_rmsd(self):
        return batch_rmsd_calculation(self.generated_pdb_paths, self.inverse_generated_pdb_paths)

    def cross_consistency_rmspd(self):
        return batch_rmspd_from_pdb_paths(self.generated_pdb_paths, self.inverse_generated_pdb_paths)

    def cross_consistency_tm(self):
        return [run_tmalign(p1, p2) for (p1, p2) in zip(self.generated_pdb_paths, self.inverse_generated_pdb_paths)]
    
    def cross_consistency_sr(self):
        gen_seqs_dict = read_sequences_from_fasta(self.generated_fasta_path)
        inv_gen_seqs_dict = read_sequences_from_fasta(self.inverse_generated_fasta_path)
        gen_seqs = [gen_seqs_dict[k] for k in sorted(gen_seqs_dict)]
        inv_gen_seqs = [inv_gen_seqs_dict[k] for k in sorted(inv_gen_seqs_dict)]
        return [calc_sequence_recovery(s1, s2) for (s1, s2) in zip(gen_seqs, inv_gen_seqs)]


class SelfConsistencyEvaluation:
    pass