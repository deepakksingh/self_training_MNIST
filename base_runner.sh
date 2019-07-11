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


echo "batch size 5000 and epoch 10"
python base_runner.py --config ./configs/config_baseline_5000_10.yml

echo "batch size 2000 and epoch 10"
python base_runner.py --config ./configs/config_baseline_2000_10.yml

echo "batch size 1000 and epoch 10"
python base_runner.py --config ./configs/config_baseline_1000_10.yml

echo "batch size 5000 and epoch 1"
python base_runner.py --config ./configs/config_baseline_5000_1.yml

echo "batch size 2000 and epoch 1"
python base_runner.py --config ./configs/config_baseline_2000_1.yml

echo "batch size 1000 and epoch 1"
python base_runner.py --config ./configs/config_baseline_1000_1.yml
