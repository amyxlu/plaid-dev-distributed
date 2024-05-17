# for task in aav fold gb1 thermo subloc fluorescence binloc stability; do
for id in g8e83omk 7str7fhl ich20c3q uhg29zk4 13lltqha fbbrfqzk kyytc8i9 8ebs7j9h identity; do
    sbatch run_plaid.slrm $id aav stability; sleep 2
    sbatch run_plaid.slrm $id gb1 fold; sleep 2
    sbatch run_plaid.slrm $id subloc thermo; sleep 2
    sbatch run_plaid.slrm $id beta fluorescence; sleep 2 
    sbatch run_plaid.slrm $id binloc; sleep 2 
done