import time
import functools
import torch

def sync_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        sync_gpu()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        sync_gpu()
        dur = (time.perf_counter()-t0)*1000
        print(f"[TIME] {fn.__qualname__}: {dur:7.2f} ms")
        return out
    return wrapper 