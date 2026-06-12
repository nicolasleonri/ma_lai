squeue --user nicolasal97

rsync -avh results/csv/ /scratch/nicolasal97/gec_extractor/results/
diff -r results/csv/ /scratch/nicolasal97/gec_extractor/results/csv

filesystem_usage
df -h /home /scratch

$ sbatch --dependency=afterany:jobID job2.slurm