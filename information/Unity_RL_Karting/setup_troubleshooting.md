# Unity Karting Microgame Troubleshooting Log

This file contains solutions and fixes to specific problems encountered while migrating from the Gymnasium environment to the Unity Karting ML-Agents environment.

## 1. Kart Inputs (WASD) Not Registering
**Symptom:** Pressing W, A, S, D or the Arrow Keys during Play Mode did not move the player's kart, even when the `KeyboardInput` script was correctly attached.
**Cause:**
Multiple input flaws in the base `ArcadeKart.cs` and `KeyboardInput.cs` scripts from the default Unity Learn project caused inputs to be dropped or overwritten:
1. `KeyboardInput` relied on legacy InputManager mappings (like "Accelerate") which sometimes conflict if the "New Input System" is active.
2. `ArcadeKart` evaluated ALL attached `IInput` components—even disabled ones—overriding the player's keyboard input with the disconnected AI's "zero" input!
3. Input was gathered inside `FixedUpdate()` rather than `Update()`, causing the Unity physics engine to drop short keyboard presses on high-performance PCs.

**Solution:**
- **Explicit KeyCodes:** Updated `KeyboardInput.cs` to explicitly map `KeyCode.W`, `S`, `A`, `D`.
- **Input Merging & Enabled Checking:** Updated `ArcadeKart.cs -> GatherInputs()` to accumulate values instead of completely rewriting them, preventing inactive or empty input components from overwriting actual active inputs.
- **Relocate to `Update`:** Moved the invocation of `GatherInputs()` to Unity's `Update()` loop to guarantee frame-perfect captures.

## 2. Invalid `NullReference` Soft Lock
**Symptom:** Both karts successfully spawn but remain completely frozen indefinitely. The starting countdown happens, but karts never receive their `m_CanMove = true` signal.
**Cause:**
In `GameFlowManager.cs`, if the user unchecks `Auto Find Karts` in the Unity Inspector and assigns the kart manually, the internal `karts` array remains entirely `null`. The game silently fails when attempting to loop through `karts` and update physics flags.
**Solution:**
- Updated `GameFlowManager.cs` to ensure `karts = new ArcadeKart[] { playerKart }` fallback operates correctly when Auto Find is disabled.

## 3. Unity CS1612 Struct Modification Error
**Symptom:** Unity Editor locked Play Mode and refused to adopt script changes.
**Cause:**
We attempted to directly modify a struct's property natively (`Input.Accelerate = true` when `Input` returns an `InputData` struct instead of a reference). 
**Solution:**
- Extracted the struct into a local variable before updating elements. This allowed scripts to compile properly and the editor to execute the new code.

## 4. Double Camera / Physics Overlap issue
**Symptom:** Only one kart is visible immediately after duplication.
**Cause:** Unity clones objects at the exact same spatial grid transform, placing the AI kart inside the Human kart’s colliders, stalling physics on both.
**Solution:** Manually drag the cloned Kart away using the Move Tool in the Scene View to separate their wheel colliders before runtime.

## Future Steps
The AI Kart is now generating `No colliders (checkpoints) assigned to KartAgent!` logs, confirming it is executing correctly. Next steps require dragging the environment colliders into the Agent’s inspector slots to establish positional tracking for RL training.
