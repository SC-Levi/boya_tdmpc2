#!/usr/bin/env bash
set -e

# —— 代理设置 ——  
export https_proxy=http://127.0.0.1:7890  
export http_proxy=http://127.0.0.1:7890  
export all_proxy=socks5://127.0.0.1:7890  

# —— 设置ManiSkill2数据集环境变量 ——
export MS2_ASSET_DIR="/home/ubuntu/Desktop/tdmpc2/tdmpc2/data"
export PARTNET_MOBILITY_DATASET="/home/ubuntu/Desktop/tdmpc2/tdmpc2/data/partnet_mobility_faucet"

# —— 激活 Conda 环境 ——  
eval "$(conda shell.bash hook)"  
conda activate tdmpc2  

# —— 添加 tdmpc2 项目到 PYTHONPATH ——  
export PYTHONPATH="/home/ubuntu/Desktop/tdmpc2:${PYTHONPATH}"  

# —— 获取激活环境下的绝对 python 路径 ——  
PYTHON_BIN="$(which python)"  
echo "Using Python: $PYTHON_BIN"

# —— 任务列表 & MoE 配置 & 随机种子 ——  
TASKS=(lift-cube pick-cube stack-cube pick-ycb turn-faucet)  
USE_MOES=(true false)  
SEEDS=(2)

# —— 并行启动每个进程，并为每个进程生成不同的 exp_name ——  
for TASK in "${TASKS[@]}"; do
  for USE_MOE in "${USE_MOES[@]}"; do
    for SEED in "${SEEDS[@]}"; do

      EXP_NAME="maniskill_${TASK}_moe_${USE_MOE}_seed_${SEED}"

      echo "Launching: task=${TASK}, use_moe=${USE_MOE}, seed=${SEED}, exp_name=${EXP_NAME}"
      "$PYTHON_BIN" train.py \
        task="${TASK}" \
        use_moe="${USE_MOE}" \
        seed="${SEED}" \
        exp_name="${EXP_NAME}" &

    done
  done
done

# 等待所有后台进程结束  
wait  
echo "✅ All 30 Maniskill runs completed."  

