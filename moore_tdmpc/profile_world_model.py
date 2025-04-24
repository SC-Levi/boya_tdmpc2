# 禁用 Kineto，避免所有 RecordFunction 的内部断言
import os
os.environ["KINETO_ENABLED"] = "0"

import torch
import time
import argparse
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

from moore_tdmpc.world_model import MooreWorldModel

def parse_args():
    parser = argparse.ArgumentParser(description="Profile MooreWorldModel operations")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    return parser.parse_args()

def create_dummy_config(args):
    """Create a config object for MooreWorldModel"""
    class Config:
        def __init__(self):
            # Basic dimensions
            self.obs_dim = 17
            self.action_dim = 6
            self.latent_dim = 128
            self.hidden_dim = 256
            
            # Moore-specific params
            self.n_experts = 4
            self.temperature = 1.0
            self.moore_temperature = 1.0
            self.use_softmax = True
            
            # Optimization settings
            self.use_checkpoint = args.use_checkpoint
            self.use_mixed_precision = args.mixed_precision
            self.reward_top_k = 1
            
            # Device
            self.device = args.device
            
            # Debug settings
            self.debug_mode = False
            
            # Make it work with getattr
            self.__dict__["use_compile"] = False
    
    return Config()

def profile_dynamics(model, args):
    """Profile dynamics operations"""
    print("\n----- Profiling Dynamics -----")
    
    # Create test data
    batch_size = args.batch_size
    seq_len = args.seq_len
    latent_dim = model.latent_dim
    action_dim = model.action_dim
    
    # Initial latent state: [B, D]
    z = torch.randn(batch_size, latent_dim, device=args.device)
    
    # Single action: [B, A]
    a_single = torch.randn(batch_size, action_dim, device=args.device)
    
    # Action sequence: [B, T, A]
    a_seq = torch.randn(batch_size, seq_len, action_dim, device=args.device)
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = model.dynamics_step(z, a_single)
            _ = model.dynamics_vectorized(z, a_seq)
    
    # Profile single-step dynamics
    print("\nProfiling single-step dynamics...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        with record_function("dynamics_step"):
            for _ in range(10):
                next_z, weights = model.dynamics_step(z, a_single)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("dynamics_step_trace.json")
    
    # Profile vectorized dynamics
    print("\nProfiling vectorized dynamics...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        with record_function("dynamics_vectorized"):
            for _ in range(5):
                next_z_seq, weights_seq = model.dynamics_vectorized(z, a_seq)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("dynamics_vectorized_trace.json")
    
    # Benchmark timing
    print("\nBenchmarking dynamics operations...")
    # Single step
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        with torch.no_grad():
            next_z, weights = model.dynamics_step(z, a_single)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"  dynamics_step: {elapsed/50*1000:.2f} ms per call")
    
    # Vectorized
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            next_z_seq, weights_seq = model.dynamics_vectorized(z, a_seq)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"  dynamics_vectorized: {elapsed/20*1000:.2f} ms per call ({elapsed/20/seq_len*1000:.2f} ms per step)")

def profile_reward(model, args):
    """Profile reward operations"""
    print("\n----- Profiling Reward -----")
    
    # Create test data
    batch_size = args.batch_size
    seq_len = args.seq_len
    latent_dim = model.latent_dim
    action_dim = model.action_dim
    
    # Latent state: [B, D]
    z = torch.randn(batch_size, latent_dim, device=args.device)
    
    # Single action: [B, A]
    a_single = torch.randn(batch_size, action_dim, device=args.device)
    
    # Latent sequence: [B, T, D]
    z_seq = torch.randn(batch_size, seq_len, latent_dim, device=args.device)
    
    # Action sequence: [B, T, A]
    a_seq = torch.randn(batch_size, seq_len, action_dim, device=args.device)
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = model.reward(z, a_single)
    
    # Profile single state-action reward
    print("\nProfiling single state-action reward...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        with record_function("reward_single"):
            for _ in range(10):
                reward_dist, weights = model.reward(z, a_single)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("reward_single_trace.json")
    
    # Profile sequence state-action reward
    print("\nProfiling sequence state-action reward...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        with record_function("reward_sequence"):
            for _ in range(5):
                reward_dist, weights = model.reward(z_seq, a_seq)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("reward_sequence_trace.json")
    
    # Benchmark timing
    print("\nBenchmarking reward operations...")
    # Single state-action
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        with torch.no_grad():
            reward_dist, weights = model.reward(z, a_single)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"  reward (single): {elapsed/50*1000:.2f} ms per call")
    
    # Sequence
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            reward_dist, weights = model.reward(z_seq, a_seq)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"  reward (sequence): {elapsed/20*1000:.2f} ms per call ({elapsed/20/seq_len*1000:.2f} ms per item)")

def main():
    args = parse_args()
    print(f"Profiling MooreWorldModel with:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Device: {args.device}")
    print(f"  Gradient checkpointing: {args.use_checkpoint}")
    print(f"  Mixed precision: {args.mixed_precision}")
    
    # Create model
    cfg = create_dummy_config(args)
    model = MooreWorldModel(cfg).to(args.device)
    model.eval()  # Use eval mode for profiling
    
    # Profile dynamics operations
    profile_dynamics(model, args)
    
    # Profile reward operations
    profile_reward(model, args)
    
    print("\nProfiling complete. Trace files generated:")
    print("- dynamics_step_trace.json")
    print("- dynamics_vectorized_trace.json")
    print("- reward_single_trace.json")
    print("- reward_sequence_trace.json")
    print("\nView these files with chrome://tracing in Chrome browser")

if __name__ == "__main__":
    main() 