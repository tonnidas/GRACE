# Print execution time
start_time=$(date +'%Y-%m-%d %H:%M:%S %Z')
function print_time() {
    end_time=$(date +'%Y-%m-%d %H:%M:%S %Z')
    runtime=$SECONDS
    echo "Start: $start_time, End: $end_time, Duration: $((runtime/3600))h:$((runtime%3600/60))m:$((runtime%60))s"
}
trap print_time EXIT

# Navigate to current folder
cd $PBS_O_WORKDIR
pwd

# Activate Conda environment
module load use.own
conda -V
eval "$(conda shell.bash hook)"
conda activate grace
python -V

# Run python script
python train.py --dataset $dataset --drop_scheme $drop_scheme

# Local:
# rsync -rav --exclude='.git' --exclude='venv' --exclude='__pycache__' "$PWD" juit@kodiak.baylor.edu:~/workspace/
# Remote: 
# ssh juit@kodiak.baylor.edu
# cd workspace/GRACE
# qsub -q gpu -l ngpus=1,ncpus=8 -v dataset="Cora",drop_scheme="hop" kodiak.sh
# qstat -n -u juit
