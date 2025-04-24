import os
import torch
import numpy as np
from glob import glob
from pathlib import Path

def load_traj_dataset(data_dir=None, mt30_path=None, mt50_path=None, dmc_path=None):
    """
    Load trajectory datasets for multitask learning.
    
    Args:
        data_dir: Path to the root data directory containing all datasets
        mt30_path: (Legacy) Path to the Meta-World 30 task dataset
        mt80_path: (Legacy) Path to the Meta-World 80 task dataset (for mt80)
        dmc_path: (Legacy) Path to the DMC dataset
    
    Returns:
        Dictionary mapping dataset names to lists of tasks, where each task contains
        the environment and trajectory dataset.
    """
    dataset = {}
    
    # New: Use data_dir if provided
    if data_dir:
        print(f"Loading datasets from root directory: {data_dir}")
        
        # Check for dataset subdirectories
        if os.path.isdir(os.path.join(data_dir, 'mt30')):
            print(f"Found Mt-30 dataset")
            dataset['mt30'] = _load_meta_dataset(os.path.join(data_dir, 'mt30'), 30)
        
        if os.path.isdir(os.path.join(data_dir, 'mt80')):
            print(f"Found Mt-80 dataset")
            dataset['mt80'] = _load_meta_dataset(os.path.join(data_dir, 'mt80'), 80)
        
        if os.path.isdir(os.path.join(data_dir, 'dmc')):
            print(f"Found DMC dataset")
            dataset['dmc'] = _load_dmc_dataset(os.path.join(data_dir, 'dmc'))
        
        # Also check for direct task folders if it's a flattened structure
        task_dirs = [d for d in os.listdir(data_dir) if d.startswith('task_') and 
                    os.path.isdir(os.path.join(data_dir, d))]
        
        if task_dirs:
            print(f"Found {len(task_dirs)} task directories directly in data_dir")
            # Create a meta dataset with all tasks found
            tasks = []
            for task_dir in sorted(task_dirs):
                task_path = os.path.join(data_dir, task_dir)
                # Extract task index from directory name
                try:
                    task_idx = int(task_dir.split('_')[1])
                except (IndexError, ValueError):
                    task_idx = len(tasks)
                
                # Ensure we have enough slots in tasks list
                while len(tasks) <= task_idx:
                    tasks.append({'env': None, 'trajectory_dataset': None})
                
                # Load trajectory data
                traj_files = glob(os.path.join(task_path, "*.pt"))
                if traj_files:
                    trajectory_data = torch.load(traj_files[0])
                    tasks[task_idx] = {
                        'env': None,
                        'trajectory_dataset': trajectory_data
                    }
                else:
                    print(f"Warning: No trajectory files found in {task_path}")
            
            # Add tasks to dataset if we found any
            if any(task['trajectory_dataset'] is not None for task in tasks):
                dataset['data'] = tasks
        
        if not dataset:
            raise ValueError(f"No valid datasets found in {data_dir}")
        
        return dataset
    
    # Legacy: Load from specific paths if data_dir not provided
    # Load Meta-30 dataset if path is provided
    if mt30_path:
        if os.path.isdir(mt30_path):
            print(f"Loading Mt-30 dataset from {mt30_path}")
            dataset['mt30'] = _load_meta_dataset(mt30_path, 30)
        else:
            print(f"Warning: Mt-30 path {mt30_path} is not a directory")
    
    # Load Meta-80 dataset if path is provided (for MT80)
    if mt80_path:
        if os.path.isdir(mt50_path):
            print(f"Loading Mt-80 dataset from {mt80_path}")
            dataset['mt80'] = _load_meta_dataset(mt80_path, 80)
        else:
            print(f"Warning: Mt-80 path {mt80_path} is not a directory")
    
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
        num_tasks: Number of tasks to load (30 or 80)
    
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
    