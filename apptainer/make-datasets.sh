#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=<yoann.poupart@hhi-extern.fraunhofer.de>
#SBATCH --job-name=apptainer
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00

#####################################################################################

# This included file contains the definition for $LOCAL_JOB_DIR to be used locally on the node.
source "/etc/slurm/local_job_dir.sh"

# Launch the apptainer image with --nv for nvidia support. Two bind mounts are used:
# - One for the ImageNet dataset and
# - One for the results (e.g. checkpoint data that you may store in $LOCAL_JOB_DIR on the node
timeout 24h apptainer exec --nv --bind ${LOCAL_JOB_DIR}:/opt/output \
    ./apptainer/script.sif python -m scripts.make_datasets \
    --output-root /opt/output

# This command copies all results generated in $LOCAL_JOB_DIR back to the submit folder regarding the job id.
cp -r ${LOCAL_JOB_DIR} ${SLURM_SUBMIT_DIR}/${SLURM_JOB_ID}

echo "$PWD/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
