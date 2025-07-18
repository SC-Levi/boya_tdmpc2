# Prismatic: Multi-Task Model-Based Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Prismatic** is an advanced model-based reinforcement learning framework built upon TD-MPC2, featuring **Mixture of Experts (MoE)** architecture for enhanced multi-task learning capabilities. This project extends the original TD-MPC2 with sophisticated memory management, expert specialization, and runtime optimization for complex humanoid and robotic control tasks.

## 🌟 Features

### Core Capabilities
- **Multi-Task Learning**: Support for diverse robotic control tasks including locomotion and manipulation
- **Mixture of Experts (MoE)**: Dynamic expert specialization with intelligent gating mechanisms
- **Advanced Memory Management**: Runtime memory optimization with automatic cleanup and monitoring
- **State & Pixel Observations**: Flexible observation space support for various environments
- **Hierarchical Control**: Goal-conditioned planning with temporal abstractions

### Technical Highlights
- **Adaptive Expert Gating**: Temperature-controlled expert selection with entropy regularization
- **Memory Leak Prevention**: Automatic gate history cleanup and aggressive memory management
- **Real-time Monitoring**: Live memory usage tracking and process control
- **Checkpoint Management**: Automatic model saving and resumption capabilities
- **Distributed Training**: Multi-GPU support with parallel task execution

## 🏗️ Architecture

```
Prismatic Framework
├── PrismaticModel (Main Agent)
│   ├── World Model (Encoder + Dynamics)
│   ├── MoE Networks (Dynamics + Reward)
│   ├── Policy Network (Actor)
│   └── Value Networks (Critics)
├── Memory Management
│   ├── Runtime Memory Manager
│   ├── Gate History Tracking
│   └── Automatic Cleanup
└── Training Infrastructure
    ├── Online/Offline Trainers
    ├── Multi-task Support
    └── Checkpoint System
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/glue25/Multi-Task-MBRL.git
cd Multi-Task-MBRL/prismatic_model

# Create and activate conda environment
conda create -n prismatic python=3.8
conda activate prismatic

# Install dependencies
pip install -e .

# Install additional requirements
pip install torch>=1.12.0
pip install hydra-core>=1.2.0
pip install tensordict
pip install matplotlib
pip install wandb
```

### Environment Setup

For **Humanoid Environments**:
```bash
# Install HumanoidBench
cd ..  # Go to parent directory
pip install -e .
```

For **MuJoCo Environments**:
```bash
# Install MuJoCo dependencies
pip install mujoco>=2.3.0
pip install dm-control
```

## 📖 Usage

### Basic Training

Train on a single humanoid task:
```bash
cd prismatic
python train.py task=humanoid_h1hand-reach-v0 use_moe=true seed=1
```

### Multi-Task Training Script

Run comprehensive experiments with the provided script:
```bash
# Navigate to project root
cd ..
./run_humanoid.sh
```

This script will:
- Train on 4 humanoid tasks: `maze`, `pole`, `slide`, `run`
- Use seeds 37 and 42 for reproducibility
- Enable MoE architecture (`use_moe=true`)
- Automatically detect and resume from checkpoints
- Save results to structured log directories

### Configuration

Key configuration parameters in `config.yaml`:

```yaml
# MoE Settings
use_moe: true              # Enable Mixture of Experts
n_experts: 4               # Number of expert networks
use_orthogonal: true       # Orthogonal expert initialization

# MoE Dynamics
moe_tau_init: 1.8         # Initial gating temperature
moe_tau_min: 0.5          # Minimum temperature
moe_tau_max: 2.0          # Maximum temperature
moe_beta: 0.02            # Temperature adaptation rate
moe_lb_alpha: 0.03        # Load balancing coefficient

# Training
steps: 3_000_000          # Total training steps
batch_size: 256           # Batch size
lr: 3e-4                  # Learning rate
buffer_size: 1_000_000    # Replay buffer size
```

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py \
    checkpoint=logs/humanoid_h1hand-reach-v0/1/4o/step_1000000.pt \
    task=humanoid_h1hand-reach-v0 \
    eval_episodes=10
```

## 🔧 Advanced Features

### Memory Management

Prismatic includes sophisticated memory management for MoE training:

- **Automatic Cleanup**: Gate histories are cleared every update step
- **Runtime Monitoring**: Real-time memory usage tracking
- **Signal Control**: Use system signals for manual cleanup:
  ```bash
  # Clean memory
  kill -USR1 <process_id>
  
  # Check status
  kill -USR2 <process_id>
  ```

### Expert Visualization

Plot expert gating patterns:
```python
from prismatic.Prismatic_model import PrismaticModel

agent = PrismaticModel(cfg)
# After training...
agent.plot_expert_gating(
    moe_block=agent.model._dynamics,
    save_path="expert_gating.png"
)
```

### Multi-Environment Support

Supported task domains:
- **Humanoid**: `humanoid_h1hand-{task}-v0`
- **MuJoCo**: `{domain}-{task}-v0`
- **ManiSkill**: `{skill}-v0`
- **MyoSuite**: `myo-{task}`

## 📊 Experiments & Results

### Benchmark Tasks

| Task Category | Environment | Description |
|---------------|-------------|-------------|
| Locomotion | `humanoid_h1hand-run-v0` | High-speed humanoid running |
| Manipulation | `humanoid_h1hand-reach-v0` | Precise reaching tasks |
| Navigation | `humanoid_h1hand-maze-v0` | Complex maze navigation |
| Balance | `humanoid_h1hand-pole-v0` | Dynamic balance control |

### Performance Metrics

The framework tracks comprehensive metrics:
- **Task Success Rate**: Environment-specific success criteria
- **Sample Efficiency**: Steps to reach performance thresholds  
- **Expert Utilization**: Gating entropy and specialization
- **Memory Usage**: Runtime memory consumption patterns

## 🛠️ Development

### Project Structure

```
prismatic_model/
├── prismatic/                 # Main source code
│   ├── Prismatic_model.py    # Core agent implementation
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation utilities
│   ├── common/               # Shared utilities
│   │   ├── world_model.py    # World model components
│   │   ├── layers.py         # MoE layer implementations
│   │   └── memory_monitor.py # Memory management
│   ├── trainer/              # Training algorithms
│   ├── envs/                 # Environment wrappers
│   └── configs/              # Configuration files
├── run_humanoid.sh          # Experiment runner
├── setup.py                 # Package configuration
└── README.md               # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with proper testing
4. Submit a pull request

Please ensure:
- Code follows black formatting
- All tests pass
- Documentation is updated
- Memory management principles are maintained

## 📚 Citation

If you use Prismatic in your research, please cite:

```bibtex
@article{prismatic2024,
  title={Prismatic: Multi-Task Model-Based RL with Mixture of Experts},
  author={Your Team},
  journal={arXiv preprint},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- Built upon [TD-MPC2](https://www.tdmpc2.com) by Nicklas Hansen et al.
- Integrates with [HumanoidBench](https://sferrazza.cc/humanoidbench_site/) environments
- MoE implementation inspired by modern transformer architectures

## 📞 Support

For questions, issues, or contributions:
- 🐛 **Issues**: [GitHub Issues](https://github.com/glue25/Multi-Task-MBRL/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/glue25/Multi-Task-MBRL/discussions)
- 📧 **Contact**: Open an issue for direct communication

---

**Happy Learning! 🚀**
