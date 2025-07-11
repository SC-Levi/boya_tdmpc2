#!/bin/bash
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=locomotion  
#SBATCH --nodes=1  
#SBATCH --gres=gpu:8                  # 8×3090  
#SBATCH --cpus-per-task=80            # 全节点 CPU  
#SBATCH --ntasks=1  
#SBATCH --qos=8gpu  
#SBATCH --time=3-00:00:00  
#SBATCH --output=run_all.out  

set -euo pipefail  
trap 'pkill -P $$' EXIT               # 任一子进程失败则全部终止  

# --------- 环境依赖 ----------  
source ~/miniconda3/etc/profile.d/conda.sh  
conda activate tdmpc2  
export OMP_NUM_THREADS=1  
export MKL_NUM_THREADS=1  
export TORCH_NUM_THREADS=1  
export WANDB_MODE=offline  
ulimit -n 65535                       # 提高文件描述符上限  
mkdir -p logs  

# --------- 启动函数 ----------  
# $1 = GPU_ID  
# $2 = log file name  
# $3 = command  
launch() {  
  CUDA_VISIBLE_DEVICES=$1 \
  EGL_VISIBLE_DEVICES=$1 \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=$1 \
  stdbuf -oL -eL bash -c "$3" > logs/$2 2>&1 &  
}  

###############################################################################  
# 任务列表 —— 图中 7 个 Locomotion 任务  
# （请根据你代码中的实际环境名称替换下面的条目）  
ENVIRONMENTS=(  
  dog-run
  dog-stand  
  dog-trot  
  dog-walk  
  humanoid-run  
  humanoid-stand  
  humanoid-walk  
)  

SEEDS=(2 30 42)                      # 三个随机种子  
# 三种配置：  
#  - baseline (use_moe=false)  
#  - MoE with 4 experts (use_moe=true, n_experts=4)  
#  - MoE with 6 experts (use_moe=true, n_experts=6)  
CATEGORIES=(  
  "false:0"  
  "true:4"  
  "true:6"  
)  

GPUS=(0 1 2 3 4 5 6 7)               # 8 张卡  
counter=0  

for cat in "${CATEGORIES[@]}"; do  
  use_moe="${cat%%:*}"  
  n_exp="${cat#*:}"  

  for env in "${ENVIRONMENTS[@]}"; do  
    for seed in "${SEEDS[@]}"; do  

      # 平均分配到一张 GPU  
      gpu_id=${GPUS[$((counter % ${#GPUS[@]}))]}  

      # 版本标记 & 额外参数  
      if [[ "$use_moe" == "false" ]]; then  
        ver="tdmpc"  
        extra_args=""  
      else  
        ver="${n_exp}o"  
        extra_args="n_experts=${n_exp}"  
      fi  

      # exp_name & 日志名  
      exp_name="${ver}_${env}_s${seed}"  
      logfile="${ver}_${env}_s${seed}_gpu${gpu_id}.out"  

      # 组装命令  
      cmd="python -m tdmpc2.train task=${env} \
use_moe=${use_moe} \
seed=${seed} \
exp_name=${exp_name}"  
      if [[ -n "$extra_args" ]]; then  
        cmd+=" ${extra_args}"  
      fi  

      echo "Launching GPU${gpu_id}: ${cmd}"  
      launch $gpu_id $logfile "$cmd"  

      ((counter++))  
    done  
  done  
done  

# 等待所有子进程结束  
wait  
echo "🎉 All 63 Locomotion runs completed."  
