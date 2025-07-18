#!/usr/bin/env python3
"""
运行时内存管理器 - 用于在训练过程中动态清理MoE gate_history内存
支持多种清理策略，无需重启训练进程
"""

import torch
import gc
import threading
import time
import signal
import os
from typing import Optional, Dict, Any, List
import psutil
from pathlib import Path


class RuntimeMemoryManager:
    """
    运行时内存管理器，提供多种方式在训练过程中清理内存
    """
    
    def __init__(self, agent=None, log_file: Optional[str] = None):
        """
        初始化运行时内存管理器
        
        Args:
            agent: TDMPC2 agent 实例
            log_file: 日志文件路径
        """
        self.agent = agent
        self.log_file = log_file
        self.monitoring = False
        self.monitor_thread = None
        self.cleanup_stats = {
            'manual_cleanups': 0,
            'auto_cleanups': 0,
            'memory_freed_gb': 0.0,
            'last_cleanup_time': None
        }
        
        # 设置信号处理器，允许外部触发清理
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """设置信号处理器，允许通过信号触发内存清理"""
        def cleanup_handler(signum, frame):
            self.log("收到清理信号，开始清理内存...")
            self.force_cleanup_all()
            
        def stats_handler(signum, frame):
            self.log("收到状态查询信号，打印内存状态...")
            self.print_memory_status()
            
        # SIGUSR1: 触发内存清理
        # SIGUSR2: 打印内存状态
        try:
            signal.signal(signal.SIGUSR1, cleanup_handler)
            signal.signal(signal.SIGUSR2, stats_handler)
            self.log("信号处理器设置完成: kill -USR1 <pid> 清理内存, kill -USR2 <pid> 查看状态")
        except Exception as e:
            self.log(f"无法设置信号处理器: {e}")
    
    def log(self, message: str):
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(log_msg + '\n')
            except Exception as e:
                print(f"写入日志文件失败: {e}")
    
    def get_gate_history_memory_usage(self) -> Dict[str, Any]:
        """
        计算当前 gate_history 的内存使用量
        
        Returns:
            包含内存使用统计的字典
        """
        stats = {
            'dynamics_history_length': 0,
            'reward_history_length': 0,
            'total_tensors': 0,
            'estimated_memory_mb': 0.0,
            'cpu_memory_mb': 0.0
        }
        
        if not self.agent:
            return stats
            
        try:
            # 检查 dynamics MoE
            if (hasattr(self.agent.model, '_dynamics') and 
                hasattr(self.agent.model._dynamics, 'gate_history')):
                dynamics_history = self.agent.model._dynamics.gate_history
                stats['dynamics_history_length'] = len(dynamics_history)
                
                # 估算内存使用
                if dynamics_history:
                    sample_tensor = dynamics_history[0]
                    tensor_size_bytes = sample_tensor.numel() * sample_tensor.element_size()
                    total_memory_bytes = tensor_size_bytes * len(dynamics_history)
                    stats['estimated_memory_mb'] += total_memory_bytes / (1024 * 1024)
                    stats['total_tensors'] += len(dynamics_history)
            
            # 检查 reward MoE
            if (hasattr(self.agent.model, '_reward') and 
                hasattr(self.agent.model._reward, 'gate_history')):
                reward_history = self.agent.model._reward.gate_history
                stats['reward_history_length'] = len(reward_history)
                
                if reward_history:
                    sample_tensor = reward_history[0]
                    tensor_size_bytes = sample_tensor.numel() * sample_tensor.element_size()
                    total_memory_bytes = tensor_size_bytes * len(reward_history)
                    stats['estimated_memory_mb'] += total_memory_bytes / (1024 * 1024)
                    stats['total_tensors'] += len(reward_history)
                    
            # 获取进程实际CPU内存使用
            process = psutil.Process()
            stats['cpu_memory_mb'] = process.memory_info().rss / (1024 * 1024)
            
        except Exception as e:
            self.log(f"获取内存统计失败: {e}")
            
        return stats
    
    def clear_gate_history(self, target: str = 'all') -> Dict[str, int]:
        """
        清理 gate_history
        
        Args:
            target: 清理目标 ('all', 'dynamics', 'reward')
            
        Returns:
            清理的记录数量统计
        """
        cleared = {'dynamics': 0, 'reward': 0}
        
        if not self.agent:
            self.log("警告: 没有设置agent实例")
            return cleared
            
        try:
            if target in ['all', 'dynamics']:
                if (hasattr(self.agent.model, '_dynamics') and 
                    hasattr(self.agent.model._dynamics, 'gate_history')):
                    cleared['dynamics'] = len(self.agent.model._dynamics.gate_history)
                    self.agent.model._dynamics.gate_history.clear()
                    
            if target in ['all', 'reward']:
                if (hasattr(self.agent.model, '_reward') and 
                    hasattr(self.agent.model._reward, 'gate_history')):
                    cleared['reward'] = len(self.agent.model._reward.gate_history)
                    self.agent.model._reward.gate_history.clear()
                    
            total_cleared = cleared['dynamics'] + cleared['reward']
            if total_cleared > 0:
                self.cleanup_stats['manual_cleanups'] += 1
                self.cleanup_stats['last_cleanup_time'] = time.time()
                self.log(f"清理完成: dynamics={cleared['dynamics']}, reward={cleared['reward']}")
                
        except Exception as e:
            self.log(f"清理gate_history失败: {e}")
            
        return cleared
    
    def partial_clear_gate_history(self, keep_recent: int = 100) -> Dict[str, int]:
        """
        部分清理 gate_history，保留最近的记录
        
        Args:
            keep_recent: 保留最近的记录数量
            
        Returns:
            清理的记录数量统计
        """
        cleared = {'dynamics': 0, 'reward': 0}
        
        if not self.agent:
            return cleared
            
        try:
            # 清理 dynamics history
            if (hasattr(self.agent.model, '_dynamics') and 
                hasattr(self.agent.model._dynamics, 'gate_history')):
                history = self.agent.model._dynamics.gate_history
                if len(history) > keep_recent:
                    cleared['dynamics'] = len(history) - keep_recent
                    self.agent.model._dynamics.gate_history = history[-keep_recent:]
                    
            # 清理 reward history
            if (hasattr(self.agent.model, '_reward') and 
                hasattr(self.agent.model._reward, 'gate_history')):
                history = self.agent.model._reward.gate_history
                if len(history) > keep_recent:
                    cleared['reward'] = len(history) - keep_recent
                    self.agent.model._reward.gate_history = history[-keep_recent:]
                    
            total_cleared = cleared['dynamics'] + cleared['reward']
            if total_cleared > 0:
                self.log(f"部分清理完成: dynamics={cleared['dynamics']}, reward={cleared['reward']}, 保留={keep_recent}")
                
        except Exception as e:
            self.log(f"部分清理失败: {e}")
            
        return cleared
    
    def force_cleanup_all(self):
        """强制清理所有可能的内存占用"""
        before_stats = self.get_gate_history_memory_usage()
        
        # 1. 清理 gate_history
        cleared = self.clear_gate_history()
        
        # 2. 清理 MoE 辅助损失
        if self.agent and hasattr(self.agent.model, 'zero_moe_aux_loss'):
            try:
                self.agent.model.zero_moe_aux_loss()
                self.log("已重置MoE辅助损失")
            except Exception as e:
                self.log(f"重置MoE辅助损失失败: {e}")
        
        # 3. 强制垃圾回收
        collected = gc.collect()
        self.log(f"垃圾回收清理了 {collected} 个对象")
        
        # 4. 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.log("已清理CUDA缓存")
        
        after_stats = self.get_gate_history_memory_usage()
        memory_freed = before_stats['estimated_memory_mb'] - after_stats['estimated_memory_mb']
        self.cleanup_stats['memory_freed_gb'] += memory_freed / 1024
        
        self.log(f"强制清理完成，释放约 {memory_freed:.2f} MB 内存")
        
    def print_memory_status(self):
        """打印当前内存状态"""
        stats = self.get_gate_history_memory_usage()
        
        print("\n" + "="*60)
        print("📊 当前内存状态")
        print("="*60)
        print(f"Dynamics Gate History: {stats['dynamics_history_length']} 条记录")
        print(f"Reward Gate History: {stats['reward_history_length']} 条记录")
        print(f"总张量数量: {stats['total_tensors']}")
        print(f"估计内存使用: {stats['estimated_memory_mb']:.2f} MB")
        print(f"进程CPU内存: {stats['cpu_memory_mb']:.2f} MB")
        
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"GPU内存: 已分配 {gpu_allocated:.2f} GB, 已保留 {gpu_reserved:.2f} GB")
        
        print(f"\n清理统计:")
        print(f"  手动清理次数: {self.cleanup_stats['manual_cleanups']}")
        print(f"  自动清理次数: {self.cleanup_stats['auto_cleanups']}")
        print(f"  总释放内存: {self.cleanup_stats['memory_freed_gb']:.2f} GB")
        
        if self.cleanup_stats['last_cleanup_time']:
            last_cleanup = time.time() - self.cleanup_stats['last_cleanup_time']
            print(f"  上次清理: {last_cleanup:.1f} 秒前")
        
        print("="*60)
        print(f"进程ID: {os.getpid()}")
        print("命令示例:")
        print(f"  清理内存: kill -USR1 {os.getpid()}")
        print(f"  查看状态: kill -USR2 {os.getpid()}")
        print("="*60 + "\n")
    
    def start_auto_monitoring(self, interval: int = 300, memory_threshold_mb: float = 1000):
        """
        启动自动内存监控和清理
        
        Args:
            interval: 检查间隔（秒）
            memory_threshold_mb: 内存阈值（MB），超过此值自动清理
        """
        if self.monitoring:
            self.log("自动监控已在运行")
            return
            
        self.monitoring = True
        
        def monitor_loop():
            self.log(f"开始自动内存监控，间隔 {interval}s，阈值 {memory_threshold_mb}MB")
            
            while self.monitoring:
                try:
                    stats = self.get_gate_history_memory_usage()
                    
                    if stats['estimated_memory_mb'] > memory_threshold_mb:
                        self.log(f"内存使用 {stats['estimated_memory_mb']:.2f}MB 超过阈值，开始自动清理")
                        self.partial_clear_gate_history(keep_recent=200)
                        self.cleanup_stats['auto_cleanups'] += 1
                        
                        # 如果还是太高，进行完全清理
                        stats_after = self.get_gate_history_memory_usage()
                        if stats_after['estimated_memory_mb'] > memory_threshold_mb * 0.8:
                            self.log("内存仍然较高，进行完全清理")
                            self.force_cleanup_all()
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    self.log(f"监控循环出错: {e}")
                    time.sleep(interval)
            
            self.log("自动内存监控已停止")
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_auto_monitoring(self):
        """停止自动内存监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.log("自动内存监控已停止")
    
    def create_control_file(self, control_file: str = "/tmp/prismatic_memory_control"):
        """
        创建控制文件，允许通过文件系统控制内存清理
        
        Args:
            control_file: 控制文件路径
        """
        def watch_control_file():
            self.log(f"开始监控控制文件: {control_file}")
            last_mtime = 0
            
            while self.monitoring:
                try:
                    if os.path.exists(control_file):
                        current_mtime = os.path.getmtime(control_file)
                        
                        if current_mtime > last_mtime:
                            last_mtime = current_mtime
                            
                            with open(control_file, 'r') as f:
                                command = f.read().strip().lower()
                            
                            if command == 'clear_all':
                                self.log("收到clear_all命令")
                                self.force_cleanup_all()
                            elif command == 'clear_partial':
                                self.log("收到clear_partial命令")
                                self.partial_clear_gate_history()
                            elif command == 'status':
                                self.log("收到status命令")
                                self.print_memory_status()
                            elif command.startswith('clear_keep_'):
                                try:
                                    keep_num = int(command.split('_')[-1])
                                    self.log(f"收到clear_keep_{keep_num}命令")
                                    self.partial_clear_gate_history(keep_recent=keep_num)
                                except ValueError:
                                    self.log(f"无效命令: {command}")
                            
                            # 清空控制文件
                            open(control_file, 'w').close()
                    
                    time.sleep(1)
                    
                except Exception as e:
                    self.log(f"控制文件监控出错: {e}")
                    time.sleep(5)
        
        control_thread = threading.Thread(target=watch_control_file, daemon=True)
        control_thread.start()
        
        # 创建控制文件的使用说明
        help_text = f"""# TDMPC2 内存控制文件
# 进程ID: {os.getpid()}
# 
# 使用方法:
# echo "clear_all" > {control_file}      # 完全清理
# echo "clear_partial" > {control_file}  # 部分清理
# echo "clear_keep_50" > {control_file}  # 保留最近50条
# echo "status" > {control_file}         # 查看状态
"""
        
        try:
            with open(control_file + ".help", 'w') as f:
                f.write(help_text)
            self.log(f"控制文件监控已启动，帮助文件: {control_file}.help")
        except Exception as e:
            self.log(f"创建帮助文件失败: {e}")


# 便捷函数
def create_memory_manager(agent=None, auto_start: bool = True) -> RuntimeMemoryManager:
    """
    创建内存管理器的便捷函数
    
    Args:
        agent: TDMPC2 agent实例
        auto_start: 是否自动启动监控
        
    Returns:
        RuntimeMemoryManager实例
    """
    log_file = f"/tmp/prismatic_memory_{os.getpid()}.log"
    manager = RuntimeMemoryManager(agent=agent, log_file=log_file)
    
    if auto_start:
        manager.start_auto_monitoring(interval=300, memory_threshold_mb=1000)
        manager.create_control_file()
    
    manager.print_memory_status()
    return manager


# 全局内存管理器实例
_global_manager = None

def get_global_memory_manager() -> Optional[RuntimeMemoryManager]:
    """获取全局内存管理器实例"""
    return _global_manager

def set_global_memory_manager(manager: RuntimeMemoryManager):
    """设置全局内存管理器实例"""
    global _global_manager
    _global_manager = manager


if __name__ == "__main__":
    # 独立运行时的测试代码
    print("TDMPC2 运行时内存管理器")
    print("可用命令:")
    print("  创建管理器: manager = RuntimeMemoryManager()")
    print("  清理内存: manager.force_cleanup_all()")
    print("  查看状态: manager.print_memory_status()")
    print("  启动监控: manager.start_auto_monitoring()") 