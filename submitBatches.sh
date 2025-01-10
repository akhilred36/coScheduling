cd $1
files=(*.slurm)

for file in ${files[@]}; do
    sbatch $file
done
