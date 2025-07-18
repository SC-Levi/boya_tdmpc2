#!/usr/bin/env bash
# ===========================================
# 修改后的实验脚本：只训练MoE模式，使用seed=37和42
# - 训练 seed=37 和 seed=42
# - 只使用 use_moe=true 模式
# - 从 /media/levi/Singe4linux/logs 路径读取checkpoint
# - 单GPU顺序执行（GPU0）
# - 总共 8 个任务（4个任务 × 2个seed）
# =========================================== 

# 移除 set -e 避免背景进程错误导致脚本提前退出
# set -e 
# 调试模式：显示执行的命令
# set -x

# 激活环境
eval "$(conda shell.bash hook)"
conda activate human

# 设置代理
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890

# 实验配置
EXPERIMENT_SEEDS=(37 42)              # 训练seed=37和42
tasks=(maze pole slide run)           # 4个基础任务
use_moe=true                          # 只使用MoE模式
exp_name="4o"                         # MoE模式对应的实验名
CHECKPOINT_BASE="/media/levi/Singe4linux/logs"  # checkpoint基础路径

# 获取脚本绝对路径，避免hydra工作目录问题
script_dir="$(cd "$(dirname "$0")" && pwd)"
# 切换到prismatic目录，确保Python模块路径正确
cd "$script_dir/prismatic"

echo "========================================"
echo "开始单GPU并行训练多个seed的MoE任务"
echo "任务列表: ${tasks[*]}"
echo "训练种子: ${EXPERIMENT_SEEDS[*]}"
echo "MoE模式: ${use_moe} (实验名: ${exp_name})"
echo "总任务数: $((${#tasks[@]} * ${#EXPERIMENT_SEEDS[@]}))"
echo "Checkpoint路径: ${CHECKPOINT_BASE}"
echo "========================================"

# 设置单GPU环境
export CUDA_VISIBLE_DEVICES=0
export EGL_VISIBLE_DEVICES=0  
export MUJOCO_GL=egl
export NVIDIA_VISIBLE_DEVICES=0
export EGL_DEVICE_ID=0

# 并行启动所有任务
pids=()  # 存储所有进程PID

for t in "${tasks[@]}"; do
  full_task="humanoid_h1hand-${t}-v0"
  
  echo "=== 处理任务: ${full_task} ==="
  
  for seed in "${EXPERIMENT_SEEDS[@]}"; do
    echo "  准备启动: ${exp_name} (use_moe=${use_moe}), seed=${seed}"
    echo "  输出目录: logs/${full_task}/${seed}/${exp_name}/"
    
    # 构造checkpoint路径
    checkpoint_path="${CHECKPOINT_BASE}/${full_task}/${seed}/${exp_name}"
    
    # 检查checkpoint是否存在
    if [[ -d "$checkpoint_path" ]]; then
      echo "  找到checkpoint目录: $checkpoint_path"
      
      # 查找最新的checkpoint文件（按步数排序）
      latest_ckpt=$(find "$checkpoint_path" -name "step_*.pt" -type f -exec basename {} \; | sed 's/step_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
      if [[ -n "$latest_ckpt" ]]; then
        latest_ckpt="step_${latest_ckpt}.pt"
      fi
      
      if [[ -n "$latest_ckpt" ]]; then
        full_checkpoint_path="${checkpoint_path}/${latest_ckpt}"
        echo "  使用checkpoint: $full_checkpoint_path"
        checkpoint_arg="checkpoint=${full_checkpoint_path}"
      else
        echo "  警告: checkpoint目录存在但未找到.pt文件，从头开始训练"
        checkpoint_arg="checkpoint=null"
      fi
    else
      echo "  未找到checkpoint目录，从头开始训练"
      checkpoint_arg="checkpoint=null"
    fi
    
    # 构造训练命令参数
    train_args=(
      task="${full_task}"
      seed="${seed}"
      use_moe="${use_moe}"
      exp_name="${exp_name}"
      "$checkpoint_arg"
    )
    
    echo "  当前工作目录: $(pwd)"
    echo "  环境变量: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "  执行命令: python train.py ${train_args[*]}"
    
    # 并行启动训练（后台执行）
    (
      python train.py "${train_args[@]}"
    ) &
    
    # 记录进程PID和相关信息
    pid=$!
    pids+=($pid)
    echo "  ✓ 已启动: ${exp_name} (${full_task}, seed=${seed}), PID=${pid}"
    
    # 短暂等待确保进程启动
    sleep 2
    echo ""
  done
  
  echo "任务 ${full_task} 的所有seed都已启动"
  echo ""
done

echo "========================================"
echo "所有任务已并行启动！"
echo "启动的进程PID列表: ${pids[*]}"
echo ""
echo "监控命令："
echo "  watch -n 5 'ps aux | grep python | grep train.py'"
echo "  htop"
echo ""
echo "等待所有训练完成..."
echo "========================================"

# 等待所有后台进程完成
wait

echo "========================================"
echo "所有任务执行完成！"
echo "已完成的任务组合："
for t in "${tasks[@]}"; do
  full_task="humanoid_h1hand-${t}-v0"
  for seed in "${EXPERIMENT_SEEDS[@]}"; do
    echo "  - ${full_task} | ${exp_name} | seed=${seed}"
  done
done
echo ""
echo "训练已全部完成！"
echo "========================================" 