#!/bin/bash
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=more
#SBATCH --nodes=1
#SBATCH --gres=gpu:8                  # 8×3090
#SBATCH --cpus-per-task=80            # 全节点 CPU
#SBATCH --ntasks=1
#SBATCH --qos=8gpu
#SBATCH --time=3-00:00:00
#SBATCH --output=run_all.out

set -euo pipefail
trap 'pkill -P $$' EXIT               # 任一子进程崩溃则全部杀

# --------- env ----------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tdmpc2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export WANDB_MODE=offline
ulimit -n 65535                       # 加大文件描述符
mkdir -p logs

# --------- launch helper ----------
# $1=gpu_id  $2=logfile  $3="python …"
launch () {
  CUDA_VISIBLE_DEVICES=$1 \
  EGL_VISIBLE_DEVICES=$1 \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=$1 \
  stdbuf -oL -eL bash -c "$3" > logs/$2 2>&1 &
}

###############################################################################
#  baseline 12  —— 平均 3 条 / GPU-0‥3
###############################################################################
# GPU-0
launch 0 tdmpc_walk.out   "python -m tdmpc2.train task=humanoid_h1hand-walk-v0   use_moe=false exp_name=tdmpc_walk"
launch 0 tdmpc_reach.out  "python -m tdmpc2.train task=humanoid_h1hand-reach-v0  use_moe=false exp_name=tdmpc_reach"
launch 0 tdmpc_hurdle.out "python -m tdmpc2.train task=humanoid_h1hand-hurdle-v0 use_moe=false exp_name=tdmpc_hurdle"

# GPU-1
launch 1 tdmpc_crawl.out  "python -m tdmpc2.train task=humanoid_h1hand-crawl-v0  use_moe=false exp_name=tdmpc_crawl"
launch 1 tdmpc_maze.out   "python -m tdmpc2.train task=humanoid_h1hand-maze-v0   use_moe=false exp_name=tdmpc_maze"
launch 1 tdmpc_stair.out  "python -m tdmpc2.train task=humanoid_h1hand-stair-v0  use_moe=false exp_name=tdmpc_stair"

# GPU-2
launch 2 tdmpc_slide.out  "python -m tdmpc2.train task=humanoid_h1hand-slide-v0  use_moe=false exp_name=tdmpc_slide"
launch 2 tdmpc_pole.out   "python -m tdmpc2.train task=humanoid_h1hand-pole-v0   use_moe=false exp_name=tdmpc_pole"
launch 2 tdmpc_balance_s.out "python -m tdmpc2.train task=humanoid_h1hand-balance_simple-v0 use_moe=false exp_name=tdmpc_balance_s"

# GPU-3
launch 3 tdmpc_balance_h.out "python -m tdmpc2.train task=humanoid_h1hand-balance_hard-v0   use_moe=false exp_name=tdmpc_balance_h"
launch 3 tdmpc_stand.out     "python -m tdmpc2.train task=humanoid_h1hand-stand-v0          use_moe=false exp_name=tdmpc_stand"
launch 3 tdmpc_run.out       "python -m tdmpc2.train task=humanoid_h1hand-run-v0            use_moe=false exp_name=tdmpc_run"

###############################################################################
#  4-experts 12  —— GPU-4 & GPU-5 各 6
###############################################################################
# GPU-4
launch 6 4o_walk.out   "python -m tdmpc2.train task=humanoid_h1hand-walk-v0   use_moe=true n_experts=4 exp_name=4o_walk"
launch 7 4o_reach.out  "python -m tdmpc2.train task=humanoid_h1hand-reach-v0  use_moe=true n_experts=4 exp_name=4o_reach"
launch 4 4o_hurdle.out "python -m tdmpc2.train task=humanoid_h1hand-hurdle-v0 use_moe=true n_experts=4 exp_name=4o_hurdle"
launch 4 4o_crawl.out  "python -m tdmpc2.train task=humanoid_h1hand-crawl-v0  use_moe=true n_experts=4 exp_name=4o_crawl"
launch 7 4o_maze.out   "python -m tdmpc2.train task=humanoid_h1hand-maze-v0   use_moe=true n_experts=4 exp_name=4o_maze"
launch 7 4o_stair.out  "python -m tdmpc2.train task=humanoid_h1hand-stair-v0  use_moe=true n_experts=4 exp_name=4o_stair"

# GPU-5
launch 6 4o_slide.out     "python -m tdmpc2.train task=humanoid_h1hand-slide-v0            use_moe=true n_experts=4 exp_name=4o_slide"
launch 6 4o_pole.out      "python -m tdmpc2.train task=humanoid_h1hand-pole-v0             use_moe=true n_experts=4 exp_name=4o_pole"
launch 7 4o_balance_s.out "python -m tdmpc2.train task=humanoid_h1hand-balance_simple-v0   use_moe=true n_experts=4 exp_name=4o_balance_s"
launch 5 4o_balance_h.out "python -m tdmpc2.train task=humanoid_h1hand-balance_hard-v0     use_moe=true n_experts=4 exp_name=4o_balance_h"
launch 5 4o_stand.out     "python -m tdmpc2.train task=humanoid_h1hand-stand-v0           use_moe=true n_experts=4 exp_name=4o_stand"
launch 5 4o_run.out       "python -m tdmpc2.train task=humanoid_h1hand-run-v0             use_moe=true n_experts=4 exp_name=4o_run"


###############################################################################
wait   # 等所有子进程结束