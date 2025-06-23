WORKDIR="GRACE"
CONDA_ENV="grace"
PYTHON_SCRIPT="train.py"

KODIAK_QUEUE="gpu" # hep
N_GPU=1
N_CPU=8

if [[ "$1" == "submit" ]]; then

    JOB_NAME="$2"
    JOB_VARIABLES="conda_env=$CONDA_ENV,python_script=$PYTHON_SCRIPT,python_args=\"$3\""

    echo "Uploading repository ..."
    rsync -rav --exclude='.git' --exclude={'venv','__pycache__','kodiak_output'} "$PWD" juit@kodiak.baylor.edu:~/workspace/

    echo "Submitting job ..."
    ssh juit@kodiak.baylor.edu "cd workspace/$WORKDIR && qsub -j oe -q $KODIAK_QUEUE -l ngpus=$N_GPU,ncpus=$N_CPU -N $JOB_NAME -v $JOB_VARIABLES kodiak.sh"

    echo "Fetching job status ..."
    ssh juit@kodiak.baylor.edu "qstat -n -u juit"

elif [[ "$1" == "sync" ]]; then

    echo "Fetching job status ..."
    ssh juit@kodiak.baylor.edu "qstat -n -u juit"

    echo "Downloading output files ..."
    scp juit@kodiak.baylor.edu:~/workspace/$WORKDIR/*.o* kodiak_output

    echo "Backup output files ..."
    ssh juit@kodiak.baylor.edu "cd workspace/$WORKDIR && mkdir -p kodiak_output_backup && mv *.o* kodiak_output_backup/" || true

elif [[ "$1" == "delete" ]]; then
    JOB_ID="$2"

    echo "Deleting job ..."
    ssh juit@kodiak.baylor.edu "qdel $JOB_ID"

    echo "Fetching job status ..."
    ssh juit@kodiak.baylor.edu "qstat -n -u juit"

elif [[ "$1" == "queue" ]]; then

    echo "Fetching queue status ..."
    ssh juit@kodiak.baylor.edu "qstat -q"

    echo "Fetching gpu status ..."
    ssh juit@kodiak.baylor.edu "freegpus"

elif [[ "$1" == "upload" ]]; then

    echo "Uploading repository ..."
    rsync -rav --exclude='.git' --exclude={'venv','__pycache__','kodiak_output'} "$PWD" juit@kodiak.baylor.edu:~/workspace/

else # qsub job script

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
    conda activate $conda_env
    python -V

    # Run python script
    python $python_script $python_args

fi
