#!/bin/bash
#SBATCH --partition=unkillable                      # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=40G                             # Ask for 40 GB of RAM
#SBATCH --time=30:00                        # The job will run for 1 hour
#SBATCH -o /home/mila/d/dylan.sandfelder/slurm-%j/log.out  # Write the log on tmp1
# ----------------------

mkdir /home/mila/d/dylan.sandfelder/slurm-%j

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
pip3 install --no-cache-dir torch-geometric


# ----------------------


# 1. Load modules
#module load pytorch
#source $CONDA_ACTIVATE
#module load cuda/10

#python -c "import torch; print(torch.__version__)"
#python -c "import torch; print(torch.cuda.is_available())"

#python -c "import torch; print(torch.version.cuda)"
#nvcc --version

#export PATH=/usr/local/cuda/bin:$PATH
#export CPATH=/usr/local/cuda/include:$CPATH
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 2. Install libraries
#pip install --no-cache-dir torch-scatter==latest+cu101 torch-sparse==latest+cu101 torch-spline-conv==latest+cu101 torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
#pip install --no-cache-dir torch-geometric
#pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
#pip install torch-scatter==2.0.2 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
#pip install torch-sparse==0.4.4 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
#pip install torch-cluster==1.4.5 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
#pip install torch-spline-conv==1.1.1 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
#pip install torch-geometric

echo "Now copying data..."

# 3. Copy your dataset on the compute node
cp -r /home/mila/d/dylan.sandfelder/Cora $SLURM_TMPDIR

echo "Launching python script..."

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python3 /home/mila/d/dylan.sandfelder/ego-gnn/ego_net_main.py --input_path $SLURM_TMPDIR

cp $SLURM_TMPDIR/model.p /home/mila/d/dylan.sandfelder/slurm-%j/
