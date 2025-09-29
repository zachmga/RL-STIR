@echo off
echo Activating RL-STIR Conda environment...
call conda activate rl-stir
echo Environment activated!
echo.
echo Available commands:
echo   python scripts/setup_env.py    - Setup environment
echo   python -m envs.sim.gen         - Generate sample episodes
echo   python scripts/train_supervised.py configs/supervised_4060.yaml - Start training
echo.
cmd /k
