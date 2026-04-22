# Unity Status - 2026-04-22

This is the current source of truth for the Unity kart training work.

## Latest Operator Notes

These notes reflect the latest hands-on testing done after the first version of this document was written.

- `SoloTrainingAgent` was turned off for pooled training.
- `RandomizeTrainingStartCheckpoint` remains off.
- short smoke tests were run with both `4` and `8` agents.
- old training artifacts in `models/` and `Unity-Kart-Racing-RL/results/` were cleared before the next long-run setup.

Important interpretation from the latest smoke tests:

- if a kart goes backward later in the episode and then touches `Checkpoint (1)` or `Checkpoint (2)` out of order, that is normal early PPO behavior
- if many karts trigger out-of-order checkpoints immediately at spawn or reset, that is a scene/layout issue

Current operating conclusion:

- `4` agents is the safer long-run setup right now
- `8` agents should only be used for a long run if a fresh smoke test looks clean at spawn

## Where We Are

The project is now past the early "trainer starts but the scene breaks" phase and into real PPO training runs inside Unity ML-Agents.

The latest verified run was the `kart_debug_01` PPO run in:

- `Unity-Kart-Racing-RL/results/kart_debug_01`
- `Unity-Kart-Racing-RL/results/episode_trace.log`
- `Unity-Kart-Racing-RL/Assets/ML-Agents/Timers/MainScene_timers.json`

That run showed:

- the trainer stayed connected for about 9 to 10 minutes
- Unity and Python both stayed alive through the run
- no gameplay-side crash or fatal runtime exception in the saved training logs
- `106` episode starts
- `83` completed episode endings recorded in the trace
- `39` `completed-lap` endings
- `44` `wrong-checkpoint` endings

This means the kart is genuinely driving laps and collecting useful experience. The environment is no longer stuck at pure startup/debug mode.

## Improvements We Have Made

### Environment and scene stability

- prevented training runs from falling into normal win/lose scene flow
- stabilized checkpoint references so training does not lose its collider list mid-run
- added safer respawn behavior that projects resets onto valid track ground
- added off-track detection for cleaner training episode resets
- kept the Unity gameplay loop alive during training instead of acting like a normal player race

### Checkpoint and lap logic

- introduced explicit ordered checkpoint discovery/config through `TrainingTrackConfig`
- added support for a dedicated `EpisodeSpawnPoint`
- fixed the old "touch current checkpoint and instantly die" case by ignoring the current checkpoint when re-entered
- made checkpoint progression strict enough to detect backward or out-of-order touches
- added structured trace logging so each episode, checkpoint touch, checkpoint advance, and ending reason can be inspected afterward

### Training quality of life

- added a `SoloTrainingAgent` guard so only one training kart stays active when needed
- added configurable multi-kart spawn spacing for future scaling
- kept PPO trainer settings in repo config files instead of relying on ad-hoc commands
- confirmed the current Unity-side trainer path is ML-Agents PPO, not the older top-level Gymnasium scripts

## Problems We Solved

These are either fully solved or improved enough that they are no longer the main blocker:

- trainer startup/connectivity failures
- scene countdown/game mode interference during training
- broken checkpoint references from disappearing microgame objectives
- bad respawns into geometry or below the track
- missing or unstable ordered checkpoint assignment
- instant deaths from re-touching the same checkpoint at spawn
- poor visibility into what each episode was doing

Historical details for the earlier checkpoint fixes remain in:

- `information/Unity_RL_Karting/04_mlagents_checkpoint_fixes.md`

## Current Main Problem

The biggest remaining issue is not "the kart cannot train." It can.

The main remaining blocker is checkpoint consistency:

- some episodes still end with `reason=wrong-checkpoint`
- a few of those happen extremely early, including effectively at spawn
- that strongly suggests a scene-side spawn/trigger overlap problem, or a mismatch between the live scene used during the run and the currently saved scene layout

The checkpoint order itself currently looks correct:

- `Checkpoint`
- `Checkpoint (1)`
- `Checkpoint (2)`

The most likely causes now are:

1. `EpisodeSpawnPoint` is too close to a checkpoint trigger
2. one checkpoint trigger volume can still be touched from the spawn lane too early
3. the live scene used during the test run had slightly different placement from the saved scene now on disk

There is now an additional nuance from later smoke tests:

- not every `wrong-checkpoint` warning is bad
- later-episode warnings caused by random or backward driving are expected early in PPO
- the warnings that matter most are the ones concentrated at spawn or immediately after reset

## What To Fix Next

Before scaling up training, do these first:

1. Move `EpisodeSpawnPoint` farther back from the first valid checkpoint crossing.
2. Verify the first checkpoint physically in front of spawn is `Checkpoint` (index `0`).
3. Keep `RandomizeTrainingStartCheckpoint` off until wrong-checkpoint endings drop sharply.
4. Re-run a short PPO test and check that step-0 or very-early wrong-checkpoint endings are gone.

Practical interpretation:

- if the scene spawns cleanly and the later warnings come from agents driving badly, training can still proceed
- if the warnings mostly happen at startup, do not start a long run until spawn layout is improved

## Training Config We Are Using

Primary trainer config:

- `Unity-Kart-Racing-RL/kart_config.yaml`

Key current settings:

- trainer: `ppo`
- `batch_size: 128`
- `buffer_size: 2048`
- `learning_rate: 3.0e-4`
- `time_horizon: 64`
- `max_steps: 2_000_000`

Longer-run variant:

- `Unity-Kart-Racing-RL/kart_config_1m.yaml`

## How To Start Actual Training

From:

```powershell
D:\Abhijit\car-racing\Unity-Kart-Racing-RL
```

Run:

```powershell
mlagents-learn kart_config.yaml --run-id kart_train_01 --force
```

Then press Play in Unity.

For a longer run:

```powershell
mlagents-learn kart_config_1m.yaml --run-id kart_train_1m --force
```

These commands assume the Python environment is already activated.

## Current Recommended Long Run

If the scene currently has `4` active training karts and the smoke test looked stable:

```powershell
mlagents-learn kart_config_1m.yaml --run-id kart_4agent_1m --force
```

If the scene actually has `8` active training karts:

```powershell
mlagents-learn kart_config_1m.yaml --run-id kart_8agent_1m --force
```

Only use the `8`-agent long run if the latest smoke test did not show obvious spawn-phase instability.

## Resume And Continue Commands

If a `1M` run is stopped in the middle, continue it with the same run id:

```powershell
mlagents-learn kart_config_1m.yaml --run-id kart_4agent_1m --resume
```

or, for an `8`-agent run:

```powershell
mlagents-learn kart_config_1m.yaml --run-id kart_8agent_1m --resume
```

If a `1M` run finishes and you want to continue the same run toward `2M`, use the `2M` config with the same run id and `--resume`.

Example for a `4`-agent run:

```powershell
mlagents-learn kart_config.yaml --run-id kart_4agent_1m --resume
```

Example for an `8`-agent run:

```powershell
mlagents-learn kart_config.yaml --run-id kart_8agent_1m --resume
```

Rule of thumb:

- use `--force` for a brand-new run
- use `--resume` for continuing an interrupted run
- keep the same run id when continuing from `1M` to `2M`

## How To Scale To More AI Agents

Unity ML-Agents pools experience from every active agent that uses the same behavior name, so scaling does not need a new trainer definition.

### Scene-level scaling

To add more active training karts in the same scene:

1. set `SoloTrainingAgent = false`
2. duplicate the training kart object
3. keep the behavior name as `KartAgent` on every copy
4. keep all copies on the same ordered checkpoint list
5. start with `4` karts, not `8+`

Important:

- do not scale until the spawn/checkpoint overlap issue is reduced
- more broken agents only create more broken experience

### Recommended first scaling target

Start with:

- `4` karts in one Unity scene
- same `KartAgent` behavior
- same PPO config

After that is stable, move to:

- `8` scene agents total

Later testing refined this decision:

- `4` agents currently looks acceptable for a real long PPO run when the bad checkpoint touches happen later from random driving
- `8` agents is still more sensitive to spawn quality and should be treated as conditional, not the default safe setup

### Process-level scaling after scene stability

The bigger speed-up comes from using a built Unity executable with multiple environments:

```powershell
mlagents-learn kart_config_1m.yaml --run-id kart_train_4env --env Builds\KartTrainer\KartTrainer.exe --num-envs 4 --no-graphics --force
```

That is the better "real training" route once the scene is stable.

## Suggested Hyperparameter Scaling

When increasing active agents, increase rollout capacity too:

- move `batch_size` from `128` to `256` or `512`
- move `buffer_size` from `2048` to `4096` or `8192`
- keep `time_horizon: 64` unless debugging shows a clear reason to change it
- increase total `max_steps` because more agents consume steps much faster

## Resume Checklist

Use this order next session:

1. open `MainScene`
2. verify checkpoint order and spawn point placement
3. run a short PPO validation session
4. confirm early `wrong-checkpoint` endings are reduced
5. disable `SoloTrainingAgent`
6. duplicate to `4` training karts
7. start the first real pooled PPO run

## Pre-Run Checklist For Long Training

Before starting a long `1M` or `2M` run:

1. keep `Behavior Name = KartAgent`
2. keep `Behavior Type = Default`
3. keep `Mode = Training`
4. keep `Model = None`
5. keep `SoloTrainingAgent = off`
6. keep `RandomizeTrainingStartCheckpoint = off`
7. turn `Show Raycasts = off`
8. turn `Write Episode Trace = off` for the long run

## Cleanup Note

Before the current long-run preparation, old training artifacts were removed from:

- `models/`
- `Unity-Kart-Racing-RL/results/`

That cleanup is useful whenever the workspace is getting cluttered and a completely fresh run is desired.
