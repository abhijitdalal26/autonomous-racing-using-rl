# Project Context: Autonomous Racing using RL

**Goal**: Train an agent to autonomously navigate and complete the track in the Gymnasium `CarRacing-v3` environment using Proximal Policy Optimization (PPO).

## Technology Stack
- **Python Version**: 3.10 (managed via Miniconda environment `car-rl`)
- **Core RL Library**: `stable-baselines3[extra]`
- **Environment**: `gymnasium[box2d]`
- **Deep Learning Backend**: `torch` (CUDA 12.4 for RTX 3050 GPU)
- **Monitoring & Visualization**: `tensorboard`, `opencv-python`

## Architectural Decisions
1. **Observation Space**: The native `CarRacing-v3` top-down RGB image (96x96) is wrapped using `GrayscaleObservation`. This drops the channel dimension to 1, significantly improving computation speed without sacrificing essential track features.
2. **Temporal Awareness**: Because single frames don't communicate velocity or acceleration, we use `VecFrameStack` (n=4). The PPO model's CNN ingests a 4-channel image representing the 4 most recent grayscale timestamps.
3. **Policy Network**: We use SB3's natively integrated `CnnPolicy` configured to handle spatial feature extraction.
4. **Environment Version**: `CarRacing-v3` differs from `v2` primarily in that it correctly returns `terminated=True` when the track lap is successfully completed.
5. **Parallelism**: Training spawns 4 parallel environments (`SubprocVecEnv` or `DummyVecEnv` depending on OS multithreading stability) to gather rollouts faster.

## Project Structure
- `train.py`: The entry-point for initiating the PPO training. It configures the environment wrappers, specifies PPO hyperparameters, and mounts Callbacks (`CheckpointCallback`, `EvalCallback`).
- `infer.py`: The evaluation script. Loads a `.zip` model from `models/` and executes a deterministic episode loop while recording and saving raw RGB frames to an `.mp4` file in `videos/`.
- `models/`: Destination folder for periodic epoch checkpoints.
- `logs/`: Destination for TensorBoard metrics (`--logdir ./logs`). 
- `videos/`: Destination for `.mp4` artifacts.

## PPO Hyperparameters
- *n_steps*: 512
- *batch_size*: 128
- *n_epochs*: 10
- *learning_rate*: 3e-4
- *gamma*: 0.99
- *gae_lambda*: 0.95
- *clip_range*: 0.2
- *ent_coef*: 0.01 (Promotes exploratory actions to avoid early local-minimums)

## Training Instructions
1. Activate virtual environment: `conda activate car-rl`
2. Run standard training: `python train.py` (~1.5M steps default)
3. During run, host logs using: `tensorboard --logdir ./logs`

## Evaluation Instructions
1. Activate virtual environment: `conda activate car-rl`
2. Infer checkpoint and save video: `python infer.py --model models/ppo_carracing_final.zip`
