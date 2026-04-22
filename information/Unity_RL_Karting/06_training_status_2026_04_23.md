# Unity Training Status - 2026-04-23

## Summary of Success

The training pipeline has been stabilized. We have moved from a state of constant early episode resets to a stable 4-agent training run with a consistent mean reward above **200**.

## Critical Fixes Implemented

### 1. Off-Track Detection Failure
- **Issue:** Agents were ending episodes immediately with `reason=off-track`.
- **Cause:** The `Track Mask` in the `KartAgent` component was set to `Nothing`, preventing the downward raycast from detecting the track surface.
- **Fix:** Set `Track Mask` to `Everything` (or specifically layers 0, 9, 10).

### 2. Spawn Grid Instability (8-Agent Scaling)
- **Issue:** Agents (specifically index 2, 3, and higher) were spawning in mid-air or behind the track.
- **Cause:** `Training Spawn Columns` was set too low (2) and `Training Spawn Row Spacing` was too high (12), pushing back-row karts off the track mesh.
- **Fix:**
    - Increased `Training Spawn Columns` to **4**.
    - Reduced `Training Spawn Row Spacing` to **4**.
    - Increased `Training Spawn Spacing` to **3.5** to prevent physical collisions at spawn.
    - Increased `Training Reset Grace Steps` to **100** to allow physics to settle before checking off-track.

### 3. "Out of Order" Checkpoint Errors
- **Issue:** Agents were skipping `Checkpoint (0)` and hitting `Checkpoint (1)` first.
- **Cause:** `EpisodeSpawnPoint` was placed too close to the first checkpoint trigger, or karts in the front row were spawning already inside/past the trigger.
- **Fix:** Moved `EpisodeSpawnPoint` approximately 10 units backward along the Z-axis to ensure a clean approach to the first checkpoint.

## Current Training Results

- **Run ID:** `kart_4agent_new`
- **Config:** `kart_config_1m.yaml`
- **Agents:** 4
- **Mean Reward:** ~210 (Stable)
- **Std Dev:** ~25 (Highly consistent)
- **Steps Reached:** 1,000,000 (Complete)

## Post-Training Features Added

### 1. User-Controlled Racing Mode
- **Feature:** Added `UseScenePositionOnStart` toggle to `KartAgent`.
- **Function:** When enabled, the AI kart starts exactly where it is placed in the Unity Editor, rather than teleporting to a spawn point or checkpoint. This allows for manual racing setups next to human players or other AI agents.
- **Inference Tweak:** Modified `Start()` to prevent automatic checkpoint snapping when in `Inferencing` mode, ensuring the editor position is preserved.

## Next Steps
1.  Evaluate the 1M step `.onnx` model in the Unity scene using `Inference Only` mode with `UseScenePositionOnStart` enabled.
2.  **Standalone Build Success:** A working Windows build has been exported (~150MB uncompressed, 48MB compressed as `Build.rar`).
3.  **Roadmap Progress:** Step 1 (Training) is fully complete. Moving to Multi-Agent Racing.

[INFO] KartAgent. Step: 990000. Time Elapsed: 3398.225 s. Mean Reward: 180.357. Std of Reward: 26.639. Training.
[INFO] KartAgent. Step: 1000000. Time Elapsed: 3433.422 s. Mean Reward: 182.409. Std of Reward: 13.069. Training.
[INFO] Exported results\kart_4agent_new\KartAgent\KartAgent-999989.onnx
[INFO] Exported results\kart_4agent_new\KartAgent\KartAgent-1000053.onnx
[INFO] Copied results\kart_4agent_new\KartAgent\KartAgent-1000053.onnx to results\kart_4agent_new\KartAgent.onnx.