# Unity Single Kart Handoff

This file is the shortest path back into the Unity work.

## Goal For The Next Session

Verify that one ML-Agents kart can train correctly in `MainScene` without:

- immediate scene loss or win transitions
- bad respawns into the track
- observation truncation warnings
- Python startup failures

Do not start multi-kart duplication until this is true.

## Current Repo-Side State

The Unity project already includes these fixes:

- `KartAgent.cs`
  - sends 8 vector observations
  - uses 5 ray sensors plus speed, direction, and acceleration flag
  - resets to checkpoint using a downward raycast onto valid ground
  - clears angular velocity on reset
- `GameFlowManager.cs`
  - detects training runs
  - skips the countdown for training
  - starts race movement immediately in training mode
  - does not push training into `WinScene` or `LoseScene`
- `MainScene.unity`
  - `VectorObservationSize` set to `8`
  - `Decision Requester` present
  - checkpoints assigned to the agent

## Current Environment-Side State

On this machine, ML-Agents had a NumPy compatibility problem:

- `mlagents 1.1.0` used `np.float`
- current environment had `numpy 2.x`
- the local environment was patched so `mlagents-learn` can start again

Important:
- this compatibility patch is in the local Conda environment, not in the repo
- if training is attempted on a new machine or fresh environment, re-check this first

## Command To Resume

From:

```powershell
D:\Abhijit\car-racing\Unity-Kart-Racing-RL
```

Run:

```powershell
mlagents-learn kart_config.yaml --run-id=Kart_PPO_Test --force --timeout-wait 120
```

Then press Play in Unity.

## What Should Happen Now

- the trainer should keep running after Unity connects
- the kart should move, crash, reset, and continue
- the whole game should not end after a few seconds
- respawns should land above the track instead of inside geometry

## What Still Needs Validation

- sustained training for 10 to 15 minutes
- reward trend improving over time
- checkpoints being reached often enough to give useful learning signal
- no hidden scene logic still interfering with training episodes

## Useful Files To Check First

- `Unity-Kart-Racing-RL/Assets/Karting/Scripts/AI/KartAgent.cs`
- `Unity-Kart-Racing-RL/Assets/Karting/Scripts/GameFlowManager.cs`
- `Unity-Kart-Racing-RL/Assets/Karting/Scenes/MainScene.unity`
- `Unity-Kart-Racing-RL/kart_config.yaml`
- `Unity-Kart-Racing-RL/results/Kart_PPO_Test/run_logs/timers.json`
- `Unity-Kart-Racing-RL/results/Kart_PPO_Test/run_logs/training_status.json`

## Decision Gate Before Multi-Kart

Only move to multiple karts after all of these are true:

1. one kart can run for an extended session without scene-level interruption
2. resets are clean and repeatable
3. rewards are no longer dominated by immediate crashes
4. at least one longer run shows actual learning, not just connectivity
