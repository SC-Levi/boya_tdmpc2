import os
import torch
import numpy as np
from glob import glob
from pathlib import Path

def load_traj_dataset(meta30_path=None, meta50_path=None, dmc_path=None):
    """
    Load trajectory datasets for multitask learning.
    
    Args:
        meta30_path: Path to the Meta-World 30 task dataset
        meta50_path: Path to the Meta-World 50 task dataset (for mt80)
        dmc_path: Path to the DMC dataset
    
    Returns:
        Dictionary mapping dataset names to lists of tasks, where each task contains
        the environment and trajectory dataset.
    """
    dataset = {}
    
    # Load Meta-30 dataset if path is provided
    if meta30_path:
        if os.path.isdir(meta30_path):
            print(f"Loading Meta-30 dataset from {meta30_path}")
            dataset['mt30'] = _load_meta_dataset(meta30_path, 30)
        else:
            print(f"Warning: Meta-30 path {meta30_path} is not a directory")
    
    # Load Meta-50 dataset if path is provided (for MT80)
    if meta50_path:
        if os.path.isdir(meta50_path):
            print(f"Loading Meta-50 dataset from {meta50_path}")
            dataset['mt50'] = _load_meta_dataset(meta50_path, 50)
        else:
            print(f"Warning: Meta-50 path {meta50_path} is not a directory")
    
    # Load DMC dataset if path is provided
    if dmc_path:
        if os.path.isdir(dmc_path):
            print(f"Loading DMC dataset from {dmc_path}")
            dataset['dmc'] = _load_dmc_dataset(dmc_path)
        else:
            print(f"Warning: DMC path {dmc_path} is not a directory")
    
    return dataset

def _load_meta_dataset(path, num_tasks):
    """
    Load Meta-World dataset from path.
    
    Args:
        path: Path to the Meta-World dataset
        num_tasks: Number of tasks to load (30 or 50)
    
    Returns:
        List of tasks, where each task contains the environment and trajectory dataset
    """
    tasks = []
    for i in range(num_tasks):
        task_path = os.path.join(path, f"task_{i}")
        if os.path.isdir(task_path):
            # Load trajectory data
            traj_files = glob(os.path.join(task_path, "*.pt"))
            if traj_files:
                trajectory_data = torch.load(traj_files[0])
                # Create dummy env for now
                env = None
                task = {
                    'env': env,
                    'trajectory_dataset': trajectory_data
                }
                tasks.append(task)
            else:
                print(f"Warning: No trajectory files found for task {i} in {task_path}")
                # Add empty task to maintain indexing
                tasks.append({'env': None, 'trajectory_dataset': None})
        else:
            print(f"Warning: Task directory {task_path} does not exist")
            # Add empty task to maintain indexing
            tasks.append({'env': None, 'trajectory_dataset': None})
    
    return tasks

def _load_dmc_dataset(path):
    """
    Load DeepMind Control dataset from path.
    
    Args:
        path: Path to the DMC dataset
    
    Returns:
        List of tasks, where each task contains the environment and trajectory dataset
    """
    tasks = []
    # Get all subdirectories, each should be a task
    task_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    for task_dir in task_dirs:
        task_path = os.path.join(path, task_dir)
        # Load trajectory data
        traj_files = glob(os.path.join(task_path, "*.pt"))
        if traj_files:
            trajectory_data = torch.load(traj_files[0])
            # Create dummy env for now
            env = None
            task = {
                'env': env,
                'trajectory_dataset': trajectory_data
            }
            tasks.append(task)
        else:
            print(f"Warning: No trajectory files found for task {task_dir} in {task_path}")
            # Add empty task to maintain indexing
            tasks.append({'env': None, 'trajectory_dataset': None})
    
    return tasks 
