# Project Overview: Autonomous Racing RL

This repository has two major tracks:

1. Gymnasium `CarRacing-v3` using Stable-Baselines3.
2. Unity Karting using ML-Agents PPO.

## Current Status

### Phase 1: Gymnasium `CarRacing-v3`
- Status: completed baseline.
- Best PPO result recorded in the project docs: about `912.91` reward.
- Main training and evaluation scripts live in the repository root.
- This phase is the stable reference point for the project.

### Phase 2: Unity Karting ML-Agents
- Status: active work in progress.
- Python trainer now starts correctly and Unity connects to it.
- Scene and agent fixes are in place for:
  - checkpoint assignment
  - decision requester
  - observation size
  - training-safe race start
  - training-safe respawn
- destroyed-reference protection for inputs and checkpoints
- Single-kart learning is not validated yet.
- A longer validation run reached about `79,804` steps before manual interrupt.
- Mean reward during that run stayed around `-1.8` to `-2.4`, so the setup is running but learning quality still needs work.
- Do not duplicate into multi-kart training yet. First confirm one kart can reset cleanly, keep running, and show improving reward.

## Recommended Resume Order

1. Use the Gymnasium phase only as the known-good reference.
2. Resume Unity single-kart training first.
3. Move to multiple Unity karts only after the single kart is stable.

## Documentation Index

### Gymnasium
1. [Gym Racing Technical Details](./Gymnasium_CarRacing/01_gym_racing_details.md)

### Unity
1. [Unity Kart Setup and Training Guide](./Unity_RL_Karting/02_unity_kart_guide.md)
2. [Unity Kart Troubleshooting Log](./Unity_RL_Karting/setup_troubleshooting.md)
3. [Unity Single Kart Handoff](./Unity_RL_Karting/03_single_kart_handoff.md)

### Planning
1. [Future RL Roadmap](./03_roadmap.md)

Last updated: 2026-04-17
