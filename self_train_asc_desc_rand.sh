#!/bin/bash
#SBATCH -A research
#SBATCH -n 40
##SBATCH --nodelist=gnode08
##SBATCH --wait-all-nodes=0
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=1024
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL

#load necessary modules
module load python/3.6.8 
module load cuda/9.0 
module load cudnn/7-cuda-9.0

#activate anaconda environment
source activate dev
echo "dev conda environment activated"

#self-training 1
echo "training begins with ascending loss order"
python self_train_cnn.py --config ./configs/config_ascending.yml
echo "training ends with ascending loss order"

#self-training 2
echo "training begins with descending loss order"
python self_train_cnn.py --config ./configs/config_descending.yml
echo "training ends with descending loss order"

#self-training 3
echo "training begins with random loss order"
python self_train_cnn.py --config ./configs/config_random.yml
echo "training ends with random loss order"