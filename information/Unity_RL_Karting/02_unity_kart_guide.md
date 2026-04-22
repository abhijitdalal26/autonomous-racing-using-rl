# Unity Kart ML-Agents Setup And Training Guide

## Current Goal

The current objective is narrow on purpose:

- verify that one kart can train correctly in Unity
- only after that, duplicate into multi-kart training

At the moment, single-kart learning is still the gate.

## Scene To Use

- Unity project folder: `Unity-Kart-Racing-RL`
- active scene: `Assets/Karting/Scenes/MainScene.unity`

## Single Kart Definition Of Done

Single-kart training is considered healthy only if:

1. Python trainer starts without import errors.
2. Unity connects and keeps running.
3. The kart can move, crash, reset, and continue for many episodes.
4. The whole game does not switch into normal win or lose scenes during training.
5. Reward trends improve over time instead of staying dominated by immediate crashes.

## Current Inspector Baseline

These are the important values the current setup expects.

### `Kart_AI`

- has `ArcadeKart`
- has `KartAgent`
- has `Behavior Parameters`
- has `Decision Requester`

### `Behavior Parameters`

- `Behavior Name`: `KartAgent`
- `Behavior Type`: default
- `Vector Observation Size`: `8`
- action space: discrete
- branch sizes: `3` and `2`

### `Decision Requester`

- `Decision Period`: `5`
- `Take Actions Between Decisions`: enabled

### `KartAgent`

- 5 sensor transforms assigned
- checkpoint colliders assigned in order
- `CheckpointMask` points to the checkpoint layer
- reward values currently used:
  - `HitPenalty = -1`
  - `PassCheckpointReward = 1`
  - `TowardsCheckpointReward = 0.1`
  - `SpeedReward = 0.05`
  - `AccelerationReward = 0.05`

## Important Training-Side Code Behavior

The repo-side scripts now do two important things for RL training:

### `GameFlowManager.cs`

- skips the normal race countdown when a training kart is present
- starts movement immediately in training mode
- does not drive training into `WinScene` or `LoseScene`

### `KartAgent.cs`

- emits 8 observations total
- respawns by raycasting down to valid ground near the checkpoint
- clears both linear and angular velocity during episode reset
- ignores destroyed checkpoint references instead of crashing on them

### `ArcadeKart.cs`

- refreshes `IInput` references when needed
- skips destroyed or missing input components instead of throwing `NullReferenceException`

## Training Command

Run from:

```powershell
D:\Abhijit\car-racing\Unity-Kart-Racing-RL
```

Command:

```powershell
mlagents-learn kart_config.yaml --run-id=Kart_PPO_Test --force --timeout-wait 120
```

Short validation run:

```powershell
mlagents-learn kart_config_test.yaml --run-id=Kart_PPO_Test_Short --force --timeout-wait 120
```

Then press Play in Unity.

## What You Should See In A Good Run

- trainer starts listening
- Unity connects
- kart begins acting almost immediately
- crashes trigger new episodes instead of ending the whole game
- no observation truncation warning
- no timeout after connection
- no destroyed-reference spam from checkpoint or input components

## What You Should Not Do Yet

Do not duplicate the kart into 10 to 20 copies yet.

Multi-kart training comes later, after the single-kart run is stable and clearly learning.

## Key Files

- `Assets/Karting/Scripts/AI/KartAgent.cs`
- `Assets/Karting/Scripts/KartSystems/ArcadeKart.cs`
- `Assets/Karting/Scripts/GameFlowManager.cs`
- `Assets/Karting/Scenes/MainScene.unity`
- `kart_config.yaml`
- `kart_config_test.yaml`
