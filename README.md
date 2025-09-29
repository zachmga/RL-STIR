# RL-STIR: Reinforcement Learning for Security Threat Investigation and Response

A reinforcement learning system for automated security incident investigation and response, designed to run efficiently on consumer GPUs and scale to professional setups.

## Project Structure

```
rl-stir/
├── README.md
├── envs/                 # gym-style env + episode generator
├── ingest/               # log collectors + parsers
├── data/                 # parquet + graph edge lists (gitignored)
├── modeling/             # encoders, heads
├── rl/                   # PPO/A2C, rollout storage
├── configs/              # yaml for diff gpus
└── scripts/              # train_*.py, eval_*.py, profile_*.py
```

## Quick Start

### Environment Setup (Conda)
```bash
# Create and activate Conda environment
conda env create -f environment.yml
conda activate rl-stir

# Or use the setup script
python scripts/setup_env.py

# On Windows, you can also use:
activate_env.bat
```

### GPU Verification
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0))"
```

### Training Pipeline
1. **Supervised Pretraining**: `python scripts/train_supervised.py configs/supervised_4060.yaml`
2. **RL Training**: `python scripts/train_ppo.py configs/ppo_4060.yaml`
3. **Evaluation**: `python scripts/eval.py configs/eval.yaml ckpt=...`

## Scaling to Dual RTX 3090s
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_ppo.py configs/ppo_3090x2.yaml
```

## Features
- Multi-modal encoding (logs + graph)
- TTP detection and tamper identification
- Cost-aware investigation actions
- GPU-optimized training pipeline
- Real-world log parsing (Linux/Windows)

