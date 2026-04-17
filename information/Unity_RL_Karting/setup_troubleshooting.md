# Unity Kart Troubleshooting Log

This file records the major issues hit while moving from the Gymnasium setup to Unity ML-Agents.

## 1. Keyboard Input Was Not Reliable

### Symptom

WASD or arrow input did not move the kart consistently in Play Mode.

### Cause

- legacy input mappings were fragile
- disabled input components could still overwrite live input
- input was being gathered in `FixedUpdate()` instead of `Update()`

### Fix

- use explicit key handling in `KeyboardInput.cs`
- merge input sources instead of blindly overwriting them in `ArcadeKart.cs`
- gather inputs in `Update()`

## 2. Manual Kart Assignment Soft-Locked Movement

### Symptom

The countdown ran but the kart never received permission to move.

### Cause

`GameFlowManager.cs` could leave its internal kart array empty when `autoFindKarts` was turned off.

### Fix

Populate the array explicitly when the kart is assigned manually.

## 3. Struct Write Error Blocked Compilation

### Symptom

Unity complained about modifying a struct member directly and scripts would not compile.

### Cause

`InputData` is a struct, so direct nested mutation was invalid.

### Fix

Copy the struct into a local variable, update it, then assign it back.

## 4. Duplicated Karts Spawned On Top Of Each Other

### Symptom

A duplicated kart appeared stuck, invisible, or physically jammed.

### Cause

Unity duplicated the object at the same transform, so colliders and wheel colliders overlapped.

### Fix

Move duplicated karts apart manually before runtime.

## 5. ML-Agents Timed Out After Unity Connected

### Symptom

Python connected to Unity but training stopped with a timeout.

### Cause

- missing `Decision Requester`
- countdown delay before the agent started acting

### Fix

- attach `Decision Requester`
- use `--timeout-wait 120`
- later, update `GameFlowManager.cs` so training no longer waits on the normal countdown

## 6. Observation Size Did Not Match The Agent Code

### Symptom

Unity warned that observations were being truncated.

### Cause

`KartAgent.cs` emitted 8 observations, but the scene was configured with a smaller vector observation size.

### Fix

Set `VectorObservationSize` to `8` in `MainScene`.

## 7. ML-Agents Failed To Start With NumPy 2.x

### Symptom

`mlagents-learn` failed before training began because `np.float` no longer exists in modern NumPy.

### Cause

Installed `mlagents 1.1.0` is not fully compatible with `numpy 2.x`.

### Fix

On this machine, the local environment was patched so ML-Agents could start again.

Important note:
- this is an environment-level fix, not a repo file change
- if the environment is rebuilt elsewhere, this issue may return

## 8. Normal Game Flow Was Ending RL Sessions

### Symptom

Training started, then the game behaved like a normal race and dropped into win or lose flow instead of continuing episode resets.

### Cause

`GameFlowManager.cs` still ran countdown, objective checks, and end-scene transitions during RL training.

### Fix

`GameFlowManager.cs` now detects training mode and:

- starts the race immediately
- skips the normal countdown
- avoids loading `WinScene` and `LoseScene` during training

## 9. Resetting To Raw Checkpoint Transforms Caused Bad Respawns

### Symptom

The kart could appear in a bad position, collide instantly, or sit embedded in the track after reset.

### Cause

The agent reset logic used the checkpoint transform directly instead of finding valid ground near it.

### Fix

`KartAgent.cs` now:

- raycasts downward from above the checkpoint
- places the kart slightly above detected ground
- aligns rotation more safely to the local surface
- clears angular velocity on episode reset

## 10. Destroyed Checkpoint References Caused `MissingReferenceException`

### Symptom

Unity reported:

- `MissingReferenceException` on `UnityEngine.BoxCollider`

### Cause

The agent still held checkpoint collider references after one or more of those objects had been destroyed or invalidated.

### Fix

`KartAgent.cs` now:

- removes dead checkpoint references before using them
- refuses to reset or reward against invalid colliders
- safely falls back when checkpoint lists become invalid

## 11. Destroyed Input References Caused `NullReferenceException` In `ArcadeKart.GatherInputs`

### Symptom

Unity reported:

- `NullReferenceException: Object reference not set to an instance of an object`
- source line inside `ArcadeKart.GatherInputs()`

### Cause

`ArcadeKart` could keep stale `IInput` references and then try to call `GenerateInput()` on a destroyed or missing component.

### Fix

`ArcadeKart.cs` now:

- refreshes input references when needed
- skips null or destroyed input components
- returns cleanly if no valid input sources exist

## Current Status

Where the Unity phase stands now:

- trainer starts correctly
- Unity connects correctly
- single-kart session reaches training steps
- repo-side training flow is much more RL-friendly now

What is still not proven:

- one long single-kart run showing actual learning
- reward improvement strong enough to justify multi-kart scaling

Latest short run evidence:

- trainer reached about `79,804` steps before manual interrupt
- mean reward stayed roughly between `-1.8` and `-2.4`
- this confirms the pipeline is active
- it does not yet prove the policy is learning well
