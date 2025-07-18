import torch
import psutil
import gc
from typing import Dict, Optional


class MemoryMonitor:
    """
    Memory monitoring utility for tracking GPU and CPU memory usage.
    Helps detect memory leaks and optimize memory consumption.
    """
    
    def __init__(self, log_interval: int = 1000):
        """
        Initialize memory monitor.
        
        Args:
            log_interval: How often to log memory usage (in steps)
        """
        self.log_interval = log_interval
        self.step_count = 0
        self.peak_gpu_memory = 0
        self.peak_cpu_memory = 0
        
    def log_memory(self, step: Optional[int] = None, force: bool = False) -> Dict[str, float]:
        """
        Log current memory usage.
        
        Args:
            step: Current training step (optional)
            force: Force logging regardless of interval
            
        Returns:
            Dictionary with memory statistics
        """
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
            
        if not force and self.step_count % self.log_interval != 0:
            return {}
            
        stats = {}
        
        # GPU Memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info()
            gpu_memory_free = gpu_memory_free / 1024**3  # GB
            gpu_memory_total = gpu_memory_total / 1024**3  # GB
            gpu_memory_used = gpu_memory_total - gpu_memory_free
            
            stats.update({
                'gpu_memory_allocated_gb': gpu_memory_allocated,
                'gpu_memory_reserved_gb': gpu_memory_reserved,
                'gpu_memory_used_gb': gpu_memory_used,
                'gpu_memory_total_gb': gpu_memory_total,
                'gpu_memory_utilization_%': (gpu_memory_used / gpu_memory_total) * 100
            })
            
            self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_memory_used)
            
        # CPU Memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**3  # GB
        system_memory = psutil.virtual_memory()
        
        stats.update({
            'cpu_memory_process_gb': cpu_memory,
            'cpu_memory_system_available_gb': system_memory.available / 1024**3,
            'cpu_memory_system_used_gb': system_memory.used / 1024**3,
            'cpu_memory_system_total_gb': system_memory.total / 1024**3,
            'cpu_memory_utilization_%': system_memory.percent
        })
        
        self.peak_cpu_memory = max(self.peak_cpu_memory, cpu_memory)
        
        return stats
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage since monitoring started."""
        return {
            'peak_gpu_memory_gb': self.peak_gpu_memory,
            'peak_cpu_memory_gb': self.peak_cpu_memory
        }
    
    def cleanup_memory(self, aggressive: bool = False):
        """
        Clean up memory.
        
        Args:
            aggressive: Whether to perform aggressive cleanup
        """
        # Python garbage collection
        gc.collect()
        
        # PyTorch cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()
                
    def check_memory_leak(self, threshold_increase_gb: float = 1.0) -> bool:
        """
        Check if there's a potential memory leak.
        
        Args:
            threshold_increase_gb: Memory increase threshold in GB
            
        Returns:
            True if potential memory leak detected
        """
        current_stats = self.log_memory(force=True)
        
        if torch.cuda.is_available() and 'gpu_memory_used_gb' in current_stats:
            gpu_increase = current_stats['gpu_memory_used_gb'] - (self.peak_gpu_memory - threshold_increase_gb)
            if gpu_increase > threshold_increase_gb:
                return True
                
        cpu_increase = current_stats['cpu_memory_process_gb'] - (self.peak_cpu_memory - threshold_increase_gb)
        if cpu_increase > threshold_increase_gb:
            return True
            
        return False
    
    def format_memory_stats(self, stats: Dict[str, float]) -> str:
        """Format memory statistics for logging."""
        if not stats:
            return "No memory stats available"
            
        lines = ["Memory Usage:"]
        
        if 'gpu_memory_used_gb' in stats:
            lines.append(f"  GPU: {stats['gpu_memory_used_gb']:.2f}/{stats['gpu_memory_total_gb']:.2f} GB "
                        f"({stats['gpu_memory_utilization_%']:.1f}%)")
        
        lines.append(f"  CPU: {stats['cpu_memory_process_gb']:.2f} GB "
                    f"(System: {stats['cpu_memory_utilization_%']:.1f}%)")
        
        return "\n".join(lines) 