# Unity Kart RL Setup & Training Guide

This guide outlines the manual steps required in the Unity Editor and the subsequent AI training process.

## 1. Scene Setup Checklist
Before starting training, ensure the following are configured in the `Kart-Racing` project:
- [ ] **Track Walls**: Tagged as `Wall`.
- [ ] **Checkpoints**: At least 3-4 invisible colliders tagged as `Checkpoint` placed around the track.
- [ ] **Kart Object**: Must have `Rigidbody`, `Box Collider`, `Behavior Parameters`, and `KartAgent` (script) attached.

## 2. Agent Configuration (Inspector)
- **Behavior Parameters**:
  - `Behavior Name`: `KartAgent`
  - `Vector Action`: Discrete (Branches: 2, Sizes: 3 and 2).
- **Kart Agent Script**:
  - `Colliders`: Drag all checkpoint objects here in order.
  - `Sensors`: Assign empty transforms on the kart as observation points.
  - `Masks`: Set to detect `Wall` and `Checkpoint` tags.

## 3. Training Process
Training is executed via the Python `ml-agents` library:
```bash
# Command to start training (to be run in terminal)
mlagents-learn kart_config.yaml --run-id=Kart_PPO_01
```

## 4. Multi-Agent Training (Optimizing Speed)
To train faster, **Duplicate** the AI Kart 10-20 times in the scene. ML-Agents will synchronize the learning from all karts simultaneously into one brain.

---
*Note: The C# logic resides in `Assets/Karting/Scripts/AI/KartAgent.cs`.*
