# Unity ML-Agents Karting Bug Fixes & Checkpoint Logic

This document details several critical bug fixes made to the Unity Karting ML-Agents environment to allow stable multi-agent training.

## 1. Checkpoint Destruction Bug
**Issue:** The default checkpoints in the Karting microgame contain an `Objective` or `Coin` script that causes them to disappear or be destroyed when the player drives through them.
**Symptom:** AI karts throw `No colliders (checkpoints) assigned to KartAgent!` errors mid-training and freeze.
**Fix:** Created duplicate checkpoints named `AI_Checkpoint`. Removed the disappearing scripts from these duplicates so they remain permanently in the scene. Assigned these permanent checkpoints to the AI.

## 2. Start Line Array Offset Bug
**Issue:** The AI targeting lines point backward at spawn, and karts get stuck at the start line or instantly die.
**Symptom:** `m_CheckpointIndex` expects the Start Line to be Element 0. If Element 0 is physically located down the road, the agent believes it is out of bounds or receives a `WrongCheckpointPenalty`.
**Fix:** The `Colliders` array in the `KartAgent` inspector must strictly include the `StartFinishLine (Box Collider)` as **Element 0**, followed sequentially by the rest of the track checkpoints.

## 3. Spawn Point Instakill Bug (KartAgent.cs)
**Issue:** When karts spawn, they spawn slightly behind the start line. When they hit the gas, they immediately trigger the Start Line's `OnTriggerEnter`. Because they are already at Checkpoint 0, hitting Checkpoint 0 again causes the script to evaluate `expected == 1`, resulting in an instant `WrongCheckpointPenalty` death.
**Fix:** Modified `KartAgent.cs` inside `OnTriggerEnter`:
```csharp
if (index == m_CheckpointIndex) return; // Ignore if we hit the checkpoint we are already at!
```

## 4. Multi-Agent Physics Explosion Bug (KartAgent.cs)
**Issue:** Karts spawn inside of each other when resetting mid-track or when randomizing start locations, causing massive physics explosions that throw them outside the track boundaries.
**Fix:** Modified `GetTrainingSpawnOffset` in `KartAgent.cs` to only apply the parking-lot grid spacing at the start line (Checkpoint 0). If they crash mid-track, they now spawn perfectly in the center of the track without grid offsets.

## 5. Feedback Flash HUD Annoyance
**Issue:** The screen constantly pulses red "Game Over" flashes during training.
**Fix:** Unticked/Disabled the `FeedbackFlashHUD` script on the UI Canvas. It is not needed for AI training.

## 6. Python JSON Serialization Bug (Numpy float32)
**Issue:** ML-Agents (v1.1.0) crashes when saving `training_status.json` because it cannot serialize `numpy.float32` rewards into JSON.
**Symptom:** `TypeError: Object of type float32 is not JSON serializable` and a corrupted `training_status.json` file.
**Fix:** Patched `mlagents/trainers/training_status.py` in the conda environment to use a custom `NumpyEncoder` that converts `np.floating` to standard Python floats.

## 7. Manual JSON Reconstruction - Missing Keys
**Issue:** When manually repairing a corrupted `training_status.json`, omitting the `auxillary_file_paths` key causes a crash during the next checkpoint save.
**Symptom:** `KeyError: 'auxillary_file_paths'` in `checkpoint_manager.py` during cleanup of old checkpoints.
**Fix:** Ensure every checkpoint entry in `training_status.json` includes the `auxillary_file_paths` list, even if it only contains the `.pt` file path.

## 8. Checkpoint Layer Mask
**Issue:** Karts drive through arches but the target magenta line does not update.
**Symptom:** No rewards gained, target stays fixed on the passed checkpoint.
**Fix:** Ensure the physical arches are on a layer that matches the **Checkpoint Mask** setting in the `KartAgent` inspector (usually the "Checkpoint" layer).
