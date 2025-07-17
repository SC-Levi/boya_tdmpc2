#!/usr/bin/env python3
"""
Memory leak test script for MoE blocks.
Tests gate_history accumulation and memory usage patterns.
"""

import torch
import time
import matplotlib.pyplot as plt
from common.memory_monitor import MemoryMonitor
from common.layers import MoEBlock
from omegaconf import OmegaConf


def create_mock_cfg():
    """Create a mock configuration for testing."""
    cfg = OmegaConf.create({
        'use_moe': True,
        'n_experts': 4,
        'use_orthogonal': True,
        'steps': 200_000,
        'monitor_mem_interval': 100
    })
    return cfg


def test_moe_memory_leak():
    """Test MoEBlock for memory leaks."""
    print("🧪 Testing MoE memory leak fixes...")
    
    cfg = create_mock_cfg()
    monitor = MemoryMonitor(log_interval=100)
    
    # Create MoE block
    moe_block = MoEBlock(
        cfg=cfg,
        in_dim=64,        # z(32) + a(32) = 64
        gate_dim=64,      # z(32) + a(32) = 64 for gate input
        hidden_dims=[128, 128],
        out_dim=64,
        n_experts=4,
        use_orthogonal=True
    ).cuda()
    
    # Track memory usage over time
    memory_history = []
    steps = []
    
    print("Running MoE forward passes...")
    for step in range(1000):
        # Create random inputs with correct dimensions
        batch_size = 32
        z = torch.randn(batch_size, 32).cuda()  # 32-dim latent state
        a = torch.randn(batch_size, 32).cuda()  # 32-dim action
        
        # Forward pass (MoEBlock expects z, a as separate arguments)
        output = moe_block(z, a)
        
        # Simulate gradients
        loss = output.mean()
        loss.backward()
        
        # Clear gradients (simulating optimizer.zero_grad())
        for param in moe_block.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        # Monitor memory every 50 steps
        if step % 50 == 0:
            stats = monitor.log_memory(step=step, force=True)
            if stats:
                memory_history.append(stats['cpu_memory_process_gb'])
                steps.append(step)
                print(f"Step {step}: {monitor.format_memory_stats(stats)}")
                print(f"Gate history length: {len(moe_block.gate_history)}")
        
        # Test manual cleanup (simulating training loop)
        if step % 100 == 0 and step > 0:
            moe_block.gate_history.clear()
            monitor.cleanup_memory()
    
    # Plot memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(steps, memory_history, 'b-', linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('CPU Memory (GB)')
    plt.title('Memory Usage Over Time (with MoE fixes)')
    plt.grid(True, alpha=0.3)
    plt.savefig('moe_memory_usage.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Final analysis
    peak_memory = monitor.get_peak_memory()
    final_stats = monitor.log_memory(force=True)
    
    print("\n📊 Test Results:")
    print(f"Peak CPU Memory: {peak_memory['peak_cpu_memory_gb']:.2f} GB")
    print(f"Final CPU Memory: {final_stats['cpu_memory_process_gb']:.2f} GB")
    print(f"Final Gate History Length: {len(moe_block.gate_history)}")
    print(f"Max History Length Setting: {moe_block.max_history_length}")
    
    # Check if memory leak is present
    memory_increase = final_stats['cpu_memory_process_gb'] - memory_history[0]
    print(f"Memory Increase: {memory_increase:.2f} GB")
    
    if memory_increase > 1.0:  # More than 1GB increase
        print("❌ Potential memory leak detected!")
        return False
    else:
        print("✅ No significant memory leak detected!")
        return True


def test_gate_history_limit():
    """Test gate history length limiting."""
    print("\n🧪 Testing gate history length limiting...")
    
    cfg = create_mock_cfg()
    moe_block = MoEBlock(
        cfg=cfg,
        in_dim=64,        # z(32) + a(32) = 64
        gate_dim=64,      # z(32) + a(32) = 64 for gate input
        hidden_dims=[128, 128],
        out_dim=64,
        n_experts=4
    ).cuda()
    
    # Override the max length for testing
    moe_block.max_history_length = 100
    
    print(f"Max history length: {moe_block.max_history_length}")
    
    # Perform many forward passes
    for step in range(150):  # More than max_history_length
        z = torch.randn(16, 32).cuda()  # 32-dim latent state
        a = torch.randn(16, 32).cuda()  # 32-dim action
        output = moe_block(z, a)
        
        if step % 20 == 0:
            print(f"Step {step}: Gate history length = {len(moe_block.gate_history)}")
    
    final_length = len(moe_block.gate_history)
    print(f"Final gate history length: {final_length}")
    
    if final_length <= moe_block.max_history_length:
        print("✅ Gate history length limiting works correctly!")
        return True
    else:
        print("❌ Gate history length limiting failed!")
        return False


if __name__ == "__main__":
    print("🔍 Starting MoE memory leak tests...\n")
    
    # Test 1: Overall memory leak test
    test1_passed = test_moe_memory_leak()
    
    # Test 2: Gate history limiting test
    test2_passed = test_gate_history_limit()
    
    print("\n📋 Test Summary:")
    print(f"Memory Leak Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Gate History Limit Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! MoE memory leak fixes are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the fixes.") 