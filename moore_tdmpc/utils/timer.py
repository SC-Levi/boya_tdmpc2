import time
import functools
from typing import Callable, Dict, Any, List, Optional
import inspect
import sys
import os
import numpy as np

# Global storage for timing results
_timing_results: Dict[str, List[float]] = {}
_timing_enabled = True

def enable_timing():
    """Enable timing measurements."""
    global _timing_enabled
    _timing_enabled = True

def disable_timing():
    """Disable timing measurements."""
    global _timing_enabled
    _timing_enabled = False

def reset_stats():
    """Reset all timing statistics."""
    global _timing_results
    _timing_results = {}

def get_stats() -> Dict[str, Dict[str, float]]:
    """Return all timing statistics.
    
    Returns:
        Dict containing for each function: mean, std, min, max, count, total time
    """
    result = {}
    for name, times in _timing_results.items():
        if not times:
            continue
        times_arr = np.array(times)
        result[name] = {
            'mean': float(np.mean(times_arr)),
            'std': float(np.std(times_arr)),
            'min': float(np.min(times_arr)),
            'max': float(np.max(times_arr)),
            'count': len(times_arr),
            'total': float(np.sum(times_arr))
        }
    return result

def print_summary(file=sys.stdout, sort_by='total'):
    """Print a summary of all timing results.
    
    Args:
        file: File object to print to (default: sys.stdout)
        sort_by: Key to sort results by (default: 'total')
    """
    stats = get_stats()
    if not stats:
        print("No timing data available.", file=file)
        return
    
    # Sort stats by the specified key
    sorted_stats = sorted(stats.items(), key=lambda x: x[1][sort_by], reverse=True)
    
    # Print header
    print("\n--- Timing Results (sorted by {}) ---".format(sort_by), file=file)
    print(f"{'Function':<40} {'Count':>10} {'Total (s)':>12} {'Mean (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12} {'Std (ms)':>12}", file=file)
    print("-" * 110, file=file)
    
    # Print each function's stats
    for name, stat in sorted_stats:
        print(f"{name:<40} {stat['count']:>10d} {stat['total']:>12.3f} {stat['mean']*1000:>12.3f} {stat['min']*1000:>12.3f} {stat['max']*1000:>12.3f} {stat['std']*1000:>12.3f}", file=file)
    
    print("-" * 110, file=file)

def timed(func=None, *, name: Optional[str] = None):
    """Decorator to measure execution time of a function.
    
    Args:
        func: The function to be decorated
        name: Optional custom name for the function in timing results
    
    Returns:
        Decorated function that measures execution time
    """
    def decorator(f):
        func_name = name or f.__qualname__
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not _timing_enabled:
                return f(*args, **kwargs)
            
            start_time = time.time()
            result = f(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            if func_name not in _timing_results:
                _timing_results[func_name] = []
            _timing_results[func_name].append(elapsed_time)
            
            return result
        
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)

def timer_context(name: str):
    """Context manager for timing code blocks.
    
    Args:
        name: Name to use for the timing entry
    
    Usage:
        with timer_context("my_operation"):
            # code to time
    """
    class TimerContext:
        def __init__(self, timer_name):
            self.timer_name = timer_name
            
        def __enter__(self):
            self.start_time = time.time() if _timing_enabled else None
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if _timing_enabled and self.start_time is not None:
                elapsed_time = time.time() - self.start_time
                if self.timer_name not in _timing_results:
                    _timing_results[self.timer_name] = []
                _timing_results[self.timer_name].append(elapsed_time)
    
    return TimerContext(name) 