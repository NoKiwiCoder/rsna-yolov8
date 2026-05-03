#!/bin/bash
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --job-name=MIA                            # create a short name for your job
#SBATCH --nodes=1                                 # node count
#SBATCH --gpus=1                                  # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=6:00:00                           # total run time limit (HH:MM:SS)
#SBATCH --partition=normal                        # partition(large/normal/cpu) where you submit
#SBATCH --account=mscaisuperpod                   # only require for multiple projects

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module purge                     # clear environment modules inherited from submission
module load Anaconda3/2023.09-0  # load the exact modules required
# 在 module load 之后、conda activate 之前添加
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate mia

/home/cwangeu/.conda/envs/mia/bin/python ./rsna-yolov8/train_lite_med_yolov8m_new.py &
PYTHON_PID=$!

# /home/cwangeu/.conda/envs/mia/bin/python ./rsna-yolov8/train_yolov8_original.py

trap "kill -USR1 $PYTHON_PID; wait $PYTHON_PID" USR1
trap "kill -TERM $PYTHON_PID; wait $PYTHON_PID" TERM

wait $PYTHON_PID

# 解决编码错误问题：sed -i 's/\r$//' run.sh
# 读取.out 文件：less logs/dental_342073.out

# 查看.out 文件：
# grep "mAP50" ./logs/MIA_399475.out      看到表头
# grep " all " ./logs/MIA_399475.out      看到具体数值