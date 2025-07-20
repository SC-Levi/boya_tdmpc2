#!/usr/bin/env bash
# 移除 set -e 避免背景进程错误导致脚本提前退出
# set -e 
# 调试模式：显示执行的命令
# set -x

# ===========================================
# 配置说明：
# - 只运行 use_moe=true 的任务（4o实验）
# - 所有任务从 1M 步的 checkpoint 开始训练
# =========================================== 

# 激活环境
eval "$(conda shell.bash hook)"
conda activate human

# 设置代理
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890

# 参数列表
tasks=(maze run pole slide)
# seeds 数组将根据 logs 目录自动推断 —— 这里先留空
# shell 中空数组声明
seeds=()
use_moes=(true)  # 只运行MoE模式

# 自动force_step配置
AUTO_FORCE_STEP=true          # 是否自动从checkpoint文件名提取步数作为force_step
FORCE_STEP_OFFSET=0           # 步数偏移量（可以在提取的步数基础上加减）

# 获取脚本绝对路径，避免hydra工作目录问题
script_dir="$(cd "$(dirname "$0")" && pwd)"

# 批量启动
for t in "${tasks[@]}"; do
  full_task="humanoid_h1hand-${t}-v0"

  # 自动检测当前 task 已存在的 seed 目录，如果找不到则回退到默认 37
  log_dir="${script_dir}/logs/${full_task}"
  task_seeds=()  # 为每个任务单独声明 seeds 数组
  if [[ -d "$log_dir" ]]; then
    # 只取纯数字目录作为 seed
    mapfile -t task_seeds < <(ls "$log_dir" | grep -E '^[0-9]+$')
  fi
  # 如果仍为空则使用默认 seed 37
  if [[ ${#task_seeds[@]} -eq 0 ]]; then
    task_seeds=(37)
  fi

  echo "检测到任务 ${full_task} 的seeds: ${task_seeds[*]}"

  # 为每个seed分配固定GPU（确保同一seed的4o和tdmpc2在同一GPU）
  gpu_count=0
  for seed in "${task_seeds[@]}"; do
    # 根据seed索引分配GPU（轮换使用GPU0和GPU1）
    assigned_gpu=$((gpu_count % 2))
    
    echo "=== 任务 ${full_task}, Seed ${seed} 分配到 GPU${assigned_gpu} ==="
    
    for use_moe in "${use_moes[@]}"; do

      # 根据 use_moe 选择 exp_name 以及目标 checkpoint 步数
      if [[ "$use_moe" == "true" ]]; then
        exp_name="4o"
        target_step=500000   # 4o 目标从 1M 步模型恢复
      else
        exp_name="tdmpc2"
        target_step=1500000  # tdmpc2 目标从 1.5M 步模型恢复
      fi
      
      # 同一seed的所有实验使用相同GPU
      GPU_ID=$assigned_gpu

      # 查找 models 目录下的 checkpoint 文件（使用绝对路径）
      models_dir="${script_dir}/logs/${full_task}/${seed}/${exp_name}/models"
      checkpoint_path=""
      
      if [[ -d "$models_dir" ]]; then
        # 优先查找目标步数的 checkpoint
        target_ckpt="${models_dir}/step_${target_step}.pt"
        if [[ -f "$target_ckpt" ]]; then
          checkpoint_path="$target_ckpt"
        else
          # 如果目标步数不存在，找最新的 checkpoint
          latest_ckpt=$(ls "$models_dir"/step_*.pt 2>/dev/null | sort -V | tail -1)
          if [[ -n "$latest_ckpt" ]]; then
            checkpoint_path="$latest_ckpt"
            echo "  警告: 未找到 step_${target_step}.pt，使用最新的: $(basename "$latest_ckpt")"
          else
            echo "  错误: ${models_dir} 中没有找到任何checkpoint文件"
          fi
        fi
      else
        echo "  错误: models目录不存在: ${models_dir}"
      fi

      # 构造训练命令参数
      train_args=(
        task="${full_task}"
        seed="${seed}"
        use_moe="${use_moe}"
        exp_name="${exp_name}"
      )
      
      if [[ -n "$checkpoint_path" ]]; then
        train_args+=(checkpoint="${checkpoint_path}")
        
        # 自动从checkpoint文件名提取步数并设置为force_step
        checkpoint_filename=$(basename "${checkpoint_path}")
        if [[ "$AUTO_FORCE_STEP" == "true" ]]; then
          if [[ "$checkpoint_filename" =~ step_([0-9]+)\.pt$ ]]; then
            extracted_step="${BASH_REMATCH[1]}"
            # 应用偏移量
            final_step=$((extracted_step + FORCE_STEP_OFFSET))
            train_args+=(force_step="${final_step}")
            
            if [[ "$FORCE_STEP_OFFSET" -ne 0 ]]; then
              echo "  启动: ${exp_name}, GPU=${GPU_ID}, ckpt=${checkpoint_filename}"
              echo "        force_step=${final_step} (提取=${extracted_step} + 偏移=${FORCE_STEP_OFFSET})"
            else
              echo "  启动: ${exp_name}, GPU=${GPU_ID}, ckpt=${checkpoint_filename}, force_step=${final_step}"
            fi
          else
            echo "  启动: ${exp_name}, GPU=${GPU_ID}, ckpt=${checkpoint_filename} (无法提取步数，使用checkpoint原始步数)"
          fi
        else
          echo "  启动: ${exp_name}, GPU=${GPU_ID}, ckpt=${checkpoint_filename} (AUTO_FORCE_STEP已禁用)"
        fi
      else
        echo "  启动: ${exp_name}, GPU=${GPU_ID}, ckpt=从零开始"
      fi
      
      # 严格绑定GPU：使用子shell完全隔离环境变量，强制C+G在同一卡
      (
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        export EGL_VISIBLE_DEVICES=$GPU_ID  
        export MUJOCO_GL=egl
        export NVIDIA_VISIBLE_DEVICES=$GPU_ID
        export EGL_DEVICE_ID=$GPU_ID
        # 清除可能的全局GPU设置
        unset CUDA_DEVICE_ORDER
        
        echo "  环境变量: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, EGL_DEVICE_ID=$EGL_DEVICE_ID"
        python -m tdmpc2.train "${train_args[@]}"
      ) &
      
      # 记录启动的进程PID
      last_pid=$!
      echo "  进程PID: $last_pid"
      
      # 短暂等待确保进程启动
      sleep 1

    done
    
    # 每处理完一个seed，GPU计数器递增
    ((gpu_count++))
  done
  
  echo "任务 ${full_task} 处理完成"
done

echo "所有任务启动完成，等待进程结束..."

# 等待所有后台任务完成
wait

echo "所有进程已结束。"


