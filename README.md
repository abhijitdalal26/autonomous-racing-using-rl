# Autonomous Car Racing using Reinforcement Learning

A Proximal Policy Optimization (PPO) agent that learns to drive and complete a lap in the **Gymnasium `CarRacing-v3`** environment from raw pixel observations.

## Demo

Watch the best-trained PPO agent complete a full lap successfully:

[![PPO best model demo GIF](./images/best_model.gif)](./videos/best_model.mp4)

Unity ML-Agents run recording:

![Unity Car Racing Recording](./images/Unity-Car-Racing-Recording.gif)

Unity training timing snapshot:

![Unity Training Time](./images/unity-traning_time.png)

---

## Project Structure

```
car-racing/
├── train.py          # PPO training script with checkpoint callbacks
├── infer.py          # Evaluation & video recording script
├── models/           # Saved checkpoints (.zip) — created during training
├── logs/             # TensorBoard logs — created during training
└── videos/           # Output MP4 videos — created during inference
```

---

## Environment

| Property | Value |
|---|---|
| Environment | `CarRacing-v3` (Gymnasium) |
| Observation | Top-down RGB image 96×96 (converted to grayscale) |
| Action space | Continuous: [steering, gas, braking] |
| Termination | `terminated=True` when track completed (new in v3) |

---

## Setup

```bash
# Create conda environment
conda create -y -n car-rl python=3.10
conda activate car-rl

# Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install RL dependencies
pip install "gymnasium[box2d]" "stable-baselines3[extra]" tensorboard opencv-python
```

---

## Training

```bash
# Quick sanity-check (5 000 steps, ~1 min)
conda activate car-rl
python train.py --test

# Full training (1.5M steps on 4 parallel envs — several hours)
python train.py

# Custom parameters
python train.py --timesteps 2000000 --n-envs 4
```

### PPO Hyperparameters

| Parameter | Value | Rationale |
|---|---|----|
| Policy | `CnnPolicy` | Image observations |
| n_steps | 512 | Steps per env per rollout |
| batch_size | 128 | Mini-batch for gradient update |
| n_epochs | 10 | Passes over each rollout buffer |
| learning_rate | 3e-4 | Adam LR — reliable default |
| gamma | 0.99 | Reward discount |
| gae_lambda | 0.95 | GAE bias-variance trade-off |
| clip_range | 0.2 | PPO clipping epsilon |
| ent_coef | 0.01 | Entropy bonus for exploration |
| Frame stack | 4 | Gives agent velocity perception |

### Monitoring Training with TensorBoard

```bash
conda activate car-rl
tensorboard --logdir ./logs
# Open http://localhost:6006
```

**Track these metrics during training:**
- `rollout/ep_rew_mean` — mean episode reward (target: >900)
- `rollout/ep_len_mean` — episode length
- `train/loss` and `train/policy_gradient_loss`
- `train/value_loss`
- `train/entropy_loss` — should decrease gradually
- `train/explained_variance` — should increase toward 1.0

---

## Inference & Video Recording

```bash
# Evaluate a checkpoint and save a video
conda activate car-rl
python infer.py --model models/ppo_carracing_50000_steps.zip   # early
python infer.py --model models/ppo_carracing_300000_steps.zip  # mid
python infer.py --model models/best_model.zip                  # best
```

Videos are saved to `./videos/`.

---

## Checkpoint Strategy (for report)

| Checkpoint | Timesteps | Expected Behaviour |
|---|---|---|
| Model 1 (Early) | ~50 000 | Car goes off-road quickly |
| Model 2 (Mid) | ~300 000-400 000 | Follows track but doesn't complete |
| Model 3 (Final) | 1 000 000+ | Completes the full lap |

---

## Hardware

- GPU: NVIDIA GeForce RTX 3050 (6 GB VRAM)
- CUDA: 12.4
- PyTorch: 2.6.0+cu124
