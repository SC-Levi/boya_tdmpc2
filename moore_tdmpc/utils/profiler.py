# ===== utils/profiler.py =====
import time
import functools
from collections import defaultdict
import numpy as np
import torch
from termcolor import colored

# Global switch to enable/disable profiling
PROFILE_ON = True

# Global dictionary to store profiling statistics
PROFILE_STATS = defaultdict(lambda: {"total_time": 0, "calls": 0, "min_time": float('inf'), "max_time": 0})

def profiled(func):
    """
    Decorator to profile function execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global PROFILE_ON, PROFILE_STATS
        
        if not PROFILE_ON:
            return func(*args, **kwargs)
        
        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Calculate elapsed time
        elapsed_time = end_time - start_time
        
        # Update statistics
        PROFILE_STATS[func.__name__]["total_time"] += elapsed_time
        PROFILE_STATS[func.__name__]["calls"] += 1
        PROFILE_STATS[func.__name__]["min_time"] = min(PROFILE_STATS[func.__name__]["min_time"], elapsed_time)
        PROFILE_STATS[func.__name__]["max_time"] = max(PROFILE_STATS[func.__name__]["max_time"], elapsed_time)
        
        # Debug output for the first few calls to each function
        if PROFILE_STATS[func.__name__]["calls"] <= 3:
            print(colored(f"PROFILER: {func.__name__} executed in {elapsed_time*1000:.2f}ms (call {PROFILE_STATS[func.__name__]['calls']})", "cyan"))
        
        return result
    
    return wrapper

def print_summary():
    """
    Print a summary of profiling statistics
    """
    global PROFILE_STATS
    
    if not PROFILE_STATS:
        print(colored("No profiling data available. Make sure PROFILE_ON=True and functions are decorated with @profiled", "red"))
        return
    
    print(colored("\n--- Profiling Summary ---", "yellow", attrs=["bold"]))
    print(colored(f"{'Function':<30} {'Calls':<10} {'Avg Time (ms)':<15} {'Total Time (ms)':<15} {'Min Time (ms)':<15} {'Max Time (ms)':<15}", "yellow"))
    print(colored("-" * 100, "yellow"))
    
    # Sort by total time
    for func_name, stats in sorted(PROFILE_STATS.items(), key=lambda x: x[1]["total_time"], reverse=True):
        avg_time = stats["total_time"] / stats["calls"] * 1000 if stats["calls"] > 0 else 0
        total_time = stats["total_time"] * 1000
        min_time = stats["min_time"] * 1000
        max_time = stats["max_time"] * 1000
        
        print(f"{func_name:<30} {stats['calls']:<10} {avg_time:<15.2f} {total_time:<15.2f} {min_time:<15.2f} {max_time:<15.2f}")
    
    print(colored("-" * 100, "yellow"))
    print(colored(f"Total functions profiled: {len(PROFILE_STATS)}", "yellow"))

def benchmark_dynamics(model, batch_size=32, horizon=50, latent_dim=None, action_dim=None, iterations=10, device=None, optimize=True):
    """
    Benchmark dynamics model performance
    
    Args:
        model: 要测试的世界模型
        batch_size: 批次大小
        horizon: 预测步数
        latent_dim: 潜在状态维度，如果为None则使用模型的latent_dim
        action_dim: 动作维度，如果为None则使用模型的action_dim
        iterations: 测试重复次数
        device: 运行设备，如果为None则使用模型的device
        optimize: 是否使用优化版本(TorchScript)
        
    Returns:
        dict: 包含基准测试结果的字典
    """
    # 创建结果字典
    results = {
        "standard": {"times": [], "avg": 0, "std": 0},
        "optimized": {"times": [], "avg": 0, "std": 0}
    }
    
    # 设置参数
    latent_dim = latent_dim or model.latent_dim
    action_dim = action_dim or model.action_dim
    device = device or model.device
    
    # 确保模型处于评估模式
    model.eval()
    
    # 如果请求优化并且模型未优化，则优化模型
    if optimize and not getattr(model, "optimized_for_inference", False):
        print("优化模型用于推理...")
        model.optimize_for_inference()
    
    # 创建测试数据
    z = torch.randn(batch_size, latent_dim, device=device)
    actions = torch.randn(batch_size, horizon, action_dim, device=device)
    
    # 进行预热
    print("预热中...")
    with torch.no_grad():
        # 预热普通版本
        model.dynamics(z, actions)
        
        # 预热优化版本
        if optimize:
            model.dynamics_fully_vectorized(z, actions)
    
    # 基准测试
    print(f"开始基准测试 (batch_size={batch_size}, horizon={horizon})...")
    
    # 测试普通版本
    print("测试标准版本...")
    for i in range(iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            model.dynamics(z, actions)
            
        torch.cuda.synchronize()
        end_time = time.time()
        
        results["standard"]["times"].append(end_time - start_time)
    
    # 计算统计数据
    results["standard"]["avg"] = np.mean(results["standard"]["times"]) * 1000  # 转换为毫秒
    results["standard"]["std"] = np.std(results["standard"]["times"]) * 1000
    
    # 如果已优化，测试优化版本
    if optimize and getattr(model, "optimized_for_inference", False):
        print("测试优化版本...")
        for i in range(iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                model.dynamics_fully_vectorized(z, actions)
                
            torch.cuda.synchronize()
            end_time = time.time()
            
            results["optimized"]["times"].append(end_time - start_time)
        
        # 计算统计数据
        results["optimized"]["avg"] = np.mean(results["optimized"]["times"]) * 1000
        results["optimized"]["std"] = np.std(results["optimized"]["times"]) * 1000
    
    # 打印结果
    print("\n--- 基准测试结果 (时间单位: 毫秒) ---")
    print(f"标准版本:  {results['standard']['avg']:.2f} ± {results['standard']['std']:.2f} ms")
    
    if optimize and getattr(model, "optimized_for_inference", False):
        speedup = results["standard"]["avg"] / results["optimized"]["avg"] if results["optimized"]["avg"] > 0 else 0
        print(f"优化版本:  {results['optimized']['avg']:.2f} ± {results['optimized']['std']:.2f} ms")
        print(f"加速比:   {speedup:.2f}x")
    
    return results

def reset_stats():
    """重置所有性能统计数据"""
    global PROFILE_STATS
    PROFILE_STATS.clear()
    print(colored("PROFILER: Statistics reset", "cyan"))

def enable_profiling():
    """启用性能分析"""
    global PROFILE_ON
    PROFILE_ON = True
    print(colored("PROFILER: Profiling enabled", "green"))

def disable_profiling():
    """禁用性能分析"""
    global PROFILE_ON
    PROFILE_ON = False
    print(colored("PROFILER: Profiling disabled", "red"))

def profile_gpu_memory():
    """Returns a string with current GPU memory usage information"""
    if not torch.cuda.is_available():
        return "GPU not available"
        
    stats = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        stats.append(f"GPU {i}: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
    
    return "\n".join(stats)

def get_profiling_data():
    """Returns a copy of the profiling statistics dictionary for external analysis"""
    return dict(PROFILE_STATS)

def get_profiling_status():
    """输出当前性能分析器状态"""
    global PROFILE_ON, PROFILE_STATS
    
    status = {
        "enabled": PROFILE_ON,
        "functions_tracked": len(PROFILE_STATS),
        "total_calls_tracked": sum(stats["calls"] for stats in PROFILE_STATS.values()),
    }
    
    print(colored(f"PROFILER STATUS: Enabled={status['enabled']}, Functions={status['functions_tracked']}, Total Calls={status['total_calls_tracked']}", "cyan"))
    
    return status
# ================================ 