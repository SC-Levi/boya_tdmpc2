#!/usr/bin/env bash
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=locomotion
#SBATCH --nodes=1
#SBATCH --gres=gpu:8                # 8×3090 on one node
#SBATCH --cpus-per-task=80          # full CPU of the node
#SBATCH --ntasks=1
#SBATCH --qos=8gpu
#SBATCH --time=3-00:00:00
#SBATCH --output=run_all.out

set -euo pipefail
trap 'pkill -P $$' EXIT

# ===== 环境依赖 =====
source ~/miniconda3/etc/profile.d/conda.sh
conda activate moore
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export WANDB_MODE=offline
ulimit -n 65535
mkdir -p logs

# ===== 并发上限；按显存改就行 =====
MAX_PER_GPU=10   # ← 每块卡同时最多几个 python 进程

# Checkpoint 基础路径
CHECKPOINT_BASE="/ceph/home/songchun01/xiaoyuan/boya_tdmpc2/tdmpc2/logs"

# ======== Launch 队列实现 =========
# 为每块卡维护一个 PID 数组（用 Bash nameref 更简洁）
for i in {0..7}; do
  declare -a "PIDS$i=()"
done

launch () {
  local gpu=$1 logfile=$2 cmd=$3
  local -n arr="PIDS$gpu"        # nameref -> 对应 GPU 的 PID 数组

  # 1) 清理已结束的 PID
  local alive=()
  for pid in "${arr[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done
  arr=("${alive[@]}")

  # 2) 如果达并发上限就等
  while ((${#arr[@]} >= MAX_PER_GPU)); do
    sleep 30
    alive=()
    for pid in "${arr[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive+=("$pid")
      fi
    done
    arr=("${alive[@]}")
  done

  # 3) 真正启动
  CUDA_VISIBLE_DEVICES=$gpu \
  EGL_VISIBLE_DEVICES=$gpu \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=$gpu \
  stdbuf -oL -eL bash -c "$cmd" > "logs/$logfile" 2>&1 &

  arr+=($!)   # 记录 PID
}

# ========= Baseline (tdmpc2) - 优先使用2M步数checkpoint =========
# dog-run: 没有2M，使用1.5M
launch 0 tdmpc2_dog-run_s2_gpu0.out      "python train.py task=dog-run use_moe=false seed=2 exp_name=tdmpc2_dog-run_s2 checkpoint=${CHECKPOINT_BASE}/dog-run/2/tdmpc2_dog-run_s2/models/step_1500000.pt +force_step=1500000"
launch 0 tdmpc2_dog-run_s30_gpu0.out     "python train.py task=dog-run use_moe=false seed=30 exp_name=tdmpc2_dog-run_s30 checkpoint=${CHECKPOINT_BASE}/dog-run/30/tdmpc2_dog-run_s30/models/step_1500000.pt +force_step=1500000"
launch 1 tdmpc2_dog-run_s42_gpu1.out     "python train.py task=dog-run use_moe=false seed=42 exp_name=tdmpc2_dog-run_s42 checkpoint=${CHECKPOINT_BASE}/dog-run/42/tdmpc2_dog-run_s42/models/step_1500000.pt +force_step=1500000"

# dog-stand: 只有seed=2有2M，其他使用1.5M
launch 1 tdmpc2_dog-stand_s2_gpu1.out    "python train.py task=dog-stand use_moe=false seed=2 exp_name=tdmpc2_dog-stand_s2 checkpoint=${CHECKPOINT_BASE}/dog-stand/2/tdmpc2_dog-stand_s2/models/step_2000000.pt +force_step=2000000"
launch 2 tdmpc2_dog-stand_s30_gpu2.out   "python train.py task=dog-stand use_moe=false seed=30 exp_name=tdmpc2_dog-stand_s30 checkpoint=${CHECKPOINT_BASE}/dog-stand/30/tdmpc2_dog-stand_s30/models/step_1500000.pt +force_step=1500000"
launch 2 tdmpc2_dog-stand_s42_gpu2.out   "python train.py task=dog-stand use_moe=false seed=42 exp_name=tdmpc2_dog-stand_s42 checkpoint=${CHECKPOINT_BASE}/dog-stand/42/tdmpc2_dog-stand_s42/models/step_1500000.pt +force_step=1500000"

# dog-trot: 全部都有2M
launch 3 tdmpc2_dog-trot_s2_gpu3.out     "python train.py task=dog-trot use_moe=false seed=2 exp_name=tdmpc2_dog-trot_s2 checkpoint=${CHECKPOINT_BASE}/dog-trot/2/tdmpc2_dog-trot_s2/models/step_2000000.pt +force_step=2000000"
launch 3 tdmpc2_dog-trot_s30_gpu3.out    "python train.py task=dog-trot use_moe=false seed=30 exp_name=tdmpc2_dog-trot_s30 checkpoint=${CHECKPOINT_BASE}/dog-trot/30/tdmpc2_dog-trot_s30/models/step_2000000.pt +force_step=2000000"
launch 4 tdmpc2_dog-trot_s42_gpu4.out    "python train.py task=dog-trot use_moe=false seed=42 exp_name=tdmpc2_dog-trot_s42 checkpoint=${CHECKPOINT_BASE}/dog-trot/42/tdmpc2_dog-trot_s42/models/step_2000000.pt +force_step=2000000"

# dog-walk: 全部都有2M
launch 4 tdmpc2_dog-walk_s2_gpu4.out     "python train.py task=dog-walk use_moe=false seed=2 exp_name=tdmpc2_dog-walk_s2 checkpoint=${CHECKPOINT_BASE}/dog-walk/2/tdmpc2_dog-walk_s2/models/step_2000000.pt +force_step=2000000"
launch 5 tdmpc2_dog-walk_s30_gpu5.out    "python train.py task=dog-walk use_moe=false seed=30 exp_name=tdmpc2_dog-walk_s30 checkpoint=${CHECKPOINT_BASE}/dog-walk/30/tdmpc2_dog-walk_s30/models/step_2000000.pt +force_step=2000000"
launch 5 tdmpc2_dog-walk_s42_gpu5.out    "python train.py task=dog-walk use_moe=false seed=42 exp_name=tdmpc2_dog-walk_s42 checkpoint=${CHECKPOINT_BASE}/dog-walk/42/tdmpc2_dog-walk_s42/models/step_2000000.pt +force_step=2000000"

# humanoid-run: 全部都有2M
launch 6 tdmpc2_humanoid-run_s2_gpu6.out "python train.py task=humanoid-run use_moe=false seed=2 exp_name=tdmpc2_humanoid-run_s2 checkpoint=${CHECKPOINT_BASE}/humanoid-run/2/tdmpc2_humanoid-run_s2/models/step_2000000.pt +force_step=2000000"
launch 6 tdmpc2_humanoid-run_s30_gpu6.out "python train.py task=humanoid-run use_moe=false seed=30 exp_name=tdmpc2_humanoid-run_s30 checkpoint=${CHECKPOINT_BASE}/humanoid-run/30/tdmpc2_humanoid-run_s30/models/step_2000000.pt +force_step=2000000"
launch 7 tdmpc2_humanoid-run_s42_gpu7.out "python train.py task=humanoid-run use_moe=false seed=42 exp_name=tdmpc2_humanoid-run_s42 checkpoint=${CHECKPOINT_BASE}/humanoid-run/42/tdmpc2_humanoid-run_s42/models/step_2000000.pt +force_step=2000000"

# humanoid-stand: 只有seed=2有2M，其他缺失（添加可用的）
launch 7 tdmpc2_humanoid-stand_s2_gpu7.out "python train.py task=humanoid-stand use_moe=false seed=2 exp_name=tdmpc2_humanoid-stand_s2 checkpoint=${CHECKPOINT_BASE}/humanoid-stand/2/tdmpc2_humanoid-stand_s2/models/step_2000000.pt +force_step=2000000"
launch 0 tdmpc2_humanoid-stand_s30_gpu0.out "python train.py task=humanoid-stand use_moe=false seed=30 exp_name=tdmpc2_humanoid-stand_s30 checkpoint=${CHECKPOINT_BASE}/humanoid-stand/30/tdmpc2_humanoid-stand_s30/models/step_1500000.pt +force_step=1500000"

# humanoid-walk: 没有任何checkpoint，从头开始
launch 1 tdmpc2_humanoid-walk_s2_gpu1.out  "python train.py task=humanoid-walk use_moe=false seed=2 exp_name=tdmpc2_humanoid-walk_s2"
launch 1 tdmpc2_humanoid-walk_s30_gpu1.out "python train.py task=humanoid-walk use_moe=false seed=30 exp_name=tdmpc2_humanoid-walk_s30"
launch 2 tdmpc2_humanoid-walk_s42_gpu2.out "python train.py task=humanoid-walk use_moe=false seed=42 exp_name=tdmpc2_humanoid-walk_s42"

# ========= MoE-4 (n_experts=4) - 4o模型都是500k步 =========
launch 0 4o_dog-run_s2_gpu0.out      "python train.py task=dog-run use_moe=true n_experts=4 seed=2 exp_name=4o_dog-run_s2 checkpoint=${CHECKPOINT_BASE}/dog-run/2/4o_dog-run_s2/models/step_500000.pt +force_step=500000"
launch 0 4o_dog-run_s30_gpu0.out     "python train.py task=dog-run use_moe=true n_experts=4 seed=30 exp_name=4o_dog-run_s30 checkpoint=${CHECKPOINT_BASE}/dog-run/30/4o_dog-run_s30/models/step_500000.pt +force_step=500000"
launch 1 4o_dog-run_s42_gpu1.out     "python train.py task=dog-run use_moe=true n_experts=4 seed=42 exp_name=4o_dog-run_s42 checkpoint=${CHECKPOINT_BASE}/dog-run/42/4o_dog-run_s42/models/step_500000.pt +force_step=500000"
launch 1 4o_dog-stand_s2_gpu1.out    "python train.py task=dog-stand use_moe=true n_experts=4 seed=2 exp_name=4o_dog-stand_s2 checkpoint=${CHECKPOINT_BASE}/dog-stand/2/4o_dog-stand_s2/models/step_500000.pt +force_step=500000"
launch 2 4o_dog-stand_s30_gpu2.out   "python train.py task=dog-stand use_moe=true n_experts=4 seed=30 exp_name=4o_dog-stand_s30 checkpoint=${CHECKPOINT_BASE}/dog-stand/30/4o_dog-stand_s30/models/step_500000.pt +force_step=500000"
launch 2 4o_dog-stand_s42_gpu2.out   "python train.py task=dog-stand use_moe=true n_experts=4 seed=42 exp_name=4o_dog-stand_s42 checkpoint=${CHECKPOINT_BASE}/dog-stand/42/4o_dog-stand_s42/models/step_500000.pt +force_step=500000"
launch 3 4o_dog-trot_s2_gpu3.out     "python train.py task=dog-trot use_moe=true n_experts=4 seed=2 exp_name=4o_dog-trot_s2 checkpoint=${CHECKPOINT_BASE}/dog-trot/2/4o_dog-trot_s2/models/step_500000.pt +force_step=500000"
launch 3 4o_dog-trot_s30_gpu3.out    "python train.py task=dog-trot use_moe=true n_experts=4 seed=30 exp_name=4o_dog-trot_s30 checkpoint=${CHECKPOINT_BASE}/dog-trot/30/4o_dog-trot_s30/models/step_500000.pt +force_step=500000"
launch 4 4o_dog-trot_s42_gpu4.out    "python train.py task=dog-trot use_moe=true n_experts=4 seed=42 exp_name=4o_dog-trot_s42 checkpoint=${CHECKPOINT_BASE}/dog-trot/42/4o_dog-trot_s42/models/step_500000.pt +force_step=500000"
launch 4 4o_dog-walk_s2_gpu4.out     "python train.py task=dog-walk use_moe=true n_experts=4 seed=2 exp_name=4o_dog-walk_s2 checkpoint=${CHECKPOINT_BASE}/dog-walk/2/4o_dog-walk_s2/models/step_500000.pt +force_step=500000"
launch 5 4o_dog-walk_s30_gpu5.out    "python train.py task=dog-walk use_moe=true n_experts=4 seed=30 exp_name=4o_dog-walk_s30 checkpoint=${CHECKPOINT_BASE}/dog-walk/30/4o_dog-walk_s30/models/step_500000.pt +force_step=500000"
launch 5 4o_dog-walk_s42_gpu5.out    "python train.py task=dog-walk use_moe=true n_experts=4 seed=42 exp_name=4o_dog-walk_s42 checkpoint=${CHECKPOINT_BASE}/dog-walk/42/4o_dog-walk_s42/models/step_500000.pt +force_step=500000"
launch 6 4o_humanoid-run_s2_gpu6.out "python train.py task=humanoid-run use_moe=true n_experts=4 seed=2 exp_name=4o_humanoid-run_s2 checkpoint=${CHECKPOINT_BASE}/humanoid-run/2/4o_humanoid-run_s2/models/step_500000.pt +force_step=500000"
launch 6 4o_humanoid-run_s30_gpu6.out "python train.py task=humanoid-run use_moe=true n_experts=4 seed=30 exp_name=4o_humanoid-run_s30 checkpoint=${CHECKPOINT_BASE}/humanoid-run/30/4o_humanoid-run_s30/models/step_500000.pt +force_step=500000"
launch 7 4o_humanoid-run_s42_gpu7.out "python train.py task=humanoid-run use_moe=true n_experts=4 seed=42 exp_name=4o_humanoid-run_s42 checkpoint=${CHECKPOINT_BASE}/humanoid-run/42/4o_humanoid-run_s42/models/step_500000.pt +force_step=500000"
launch 7 4o_humanoid-stand_s2_gpu7.out "python train.py task=humanoid-stand use_moe=true n_experts=4 seed=2 exp_name=4o_humanoid-stand_s2 checkpoint=${CHECKPOINT_BASE}/humanoid-stand/2/4o_humanoid-stand_s2/models/step_500000.pt +force_step=500000"
launch 0 4o_humanoid-stand_s30_gpu0.out "python train.py task=humanoid-stand use_moe=true n_experts=4 seed=30 exp_name=4o_humanoid-stand_s30 checkpoint=${CHECKPOINT_BASE}/humanoid-stand/30/4o_humanoid-stand_s30/models/step_500000.pt +force_step=500000"

# humanoid-walk MoE: 没有checkpoint，从头开始
launch 1 4o_humanoid-walk_s2_gpu1.out  "python train.py task=humanoid-walk use_moe=true n_experts=4 seed=2 exp_name=4o_humanoid-walk_s2"
launch 1 4o_humanoid-walk_s30_gpu1.out "python train.py task=humanoid-walk use_moe=true n_experts=4 seed=30 exp_name=4o_humanoid-walk_s30"
launch 2 4o_humanoid-walk_s42_gpu2.out "python train.py task=humanoid-walk use_moe=true n_experts=4 seed=42 exp_name=4o_humanoid-walk_s42"

wait
echo "🎉  All runs completed."