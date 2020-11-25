#!/bin/bash
#SBATCH --partition=unkillable                # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=40G                             # Ask for 40 GB of RAM
#SBATCH --time=30:00                          # The job will run for 0.5 hours
#SBATCH -o ./slurm-%j/log.out	              # Write the log on tmp1
# ----------------------

rm ~/ego-gnn/model?.p

echo "Loading modules..."

module load python/3.7
module load cuda/10.1

echo "Updating paths..."

export PATH=/usr/local/cuda-10.1/bin:$PATH
export CPATH=/usr/local/cuda-10.1/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH

echo "Installing pip packages..."

pip3 install --no-cache-dir torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --no-cache-dir torch-scatter==latest+cu101 torch-sparse==latest+cu101 torch-spline-conv==latest+cu101 torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip3 install --no-cache-dir torch-geometric==1.6.1

pip3 install ogb

pip3 install wandb

pip3 install -U scikit-learn

#pip3 install plotly

# ----------------------

echo "Now copying data..."

dataLocation=$(python3 ~/ego-gnn/getLocation.py)
echo "Getting data from $HOME$dataLocation"
# 3. Copy your dataset on the compute node
cp -r "$HOME$dataLocation" $SLURM_TMPDIR

echo "Logging in to W and B"

wandb login 393993fdf806bf8728a51ec00b7d1a114ce36a42

echo "Presenting current configs:"

cat ~/ego-gnn/EGONETCONFIG.py

echo "Launching python script..."


# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python3 ~/ego-gnn/ego_net_main.py --input_path $SLURM_TMPDIR
