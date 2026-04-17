# Gymnasium CarRacing Technical Details

## Status

This phase is the stable reference baseline for the project.

- Environment: `gymnasium` `CarRacing-v3`
- Main solved setup: PPO
- Best result documented in the project: about `912.91` reward
- Current focus is no longer Gymnasium experimentation; Unity is the active phase

## Purpose Of This Phase

The Gymnasium work established a working end-to-end reinforcement learning pipeline before moving into Unity.
It proved:

- environment wrapping
- image-based policy learning
- checkpointing
- TensorBoard monitoring
- evaluation video generation

## Core Stack

- Python: `3.10`
- RL library: `stable-baselines3[extra]`
- Environment: `gymnasium[box2d]`
- Deep learning: `torch`
- Monitoring: `tensorboard`
- Video export: `opencv-python`

## Files In Active Use

- `train.py`
  - PPO trainer for `CarRacing-v3`
- `infer.py`
  - PPO evaluation and video export
- `train_sac.py`
  - SAC experiment script
- `infer_sac.py`
  - SAC evaluation script
- `models/`
  - saved checkpoints
- `logs/`
  - TensorBoard logs
- `videos/`
  - exported evaluation videos

## PPO Baseline Design

The solved PPO setup uses:

- `CnnPolicy`
- grayscale observations
- 4-frame stack
- vectorized environments
- TensorBoard logging
- checkpoint and evaluation callbacks

### Why grayscale and frame stacking were used

- grayscale reduces computation while preserving track structure
- a single image does not encode motion well
- stacking 4 frames gives the policy enough short-term temporal context to infer movement and steering effect

## PPO Hyperparameters In `train.py`

- `learning_rate = 3e-4`
- `n_steps = 512`
- `batch_size = 128`
- `n_epochs = 10`
- `gamma = 0.99`
- `gae_lambda = 0.95`
- `clip_range = 0.2`
- `ent_coef = 0.01`

These values were kept close to proven PPO defaults for image control tasks and produced the successful baseline.

## PPO Training Commands

Standard run:

```powershell
conda activate car-rl
python train.py
```

Quick smoke test:

```powershell
conda activate car-rl
python train.py --test
```

TensorBoard:

```powershell
conda activate car-rl
tensorboard --logdir ./logs
```

## PPO Evaluation

```powershell
conda activate car-rl
python infer.py --model models/best_model.zip
```

The script runs one deterministic episode and writes an MP4 to `videos/`.

## SAC Experiment Track

There is also a SAC branch in the repo:

- trainer: `train_sac.py`
- evaluator: `infer_sac.py`

This was exploratory work, not the main solved baseline. The documented successful result still belongs to PPO.

## Useful Gymnasium Artifacts

- `models/best_model.zip`
  - best PPO checkpoint saved by evaluation callback
- `models/ppo_carracing_final.zip`
  - final PPO save from the long run
- `images/`
  - plots and visuals used for reporting
- `logs/PPO_CarRacing_*`
  - TensorBoard event files

## Why This Matters To The Unity Phase

The Gymnasium phase is the project sanity check:

- the RL workflow works
- the machine can train image-based policies
- the repo already has a proven single-agent baseline

When Unity behaves badly, compare it against this phase as the known-good control case.
