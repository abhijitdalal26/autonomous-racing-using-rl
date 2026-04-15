"""
train.py — PPO agent for CarRacing-v3 (Gymnasium)
================================================
Uses Stable-Baselines3 PPO with a CNN policy.
Checkpoints are saved at regular intervals so you can later
compare early, mid, and final training stages.

Usage:
  # Quick sanity check (5 000 steps, saves a checkpoint)
  conda run -n car-rl python train.py --test

  # Full training (~1.5M steps on 4 parallel envs)
  conda run -n car-rl python train.py

TensorBoard:
  conda run -n car-rl tensorboard --logdir ./logs
"""

import os
import sys
import argparse
import torch
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    VecMonitor,
)
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)


# ──────────────────────────────────────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────────────────────────────────────

def make_env(rank: int, seed: int = 42):
    """Return a factory for a single CarRacing-v3 environment."""
    def _init():
        env = gym.make("CarRacing-v3", continuous=True)
        env = GrayscaleObservation(env, keep_dim=True)   # H×W×1 → less compute
        env.reset(seed=seed + rank)
        return env
    return _init


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(total_timesteps: int = 1_500_000, n_envs: int = 4, test_mode: bool = False):
    # Folders
    log_dir   = "./logs/"
    model_dir = "./models/"
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ── Print GPU info ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)} | "
              f"VRAM {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
    else:
        print("[WARN] CUDA not found – training on CPU (slow!)")

    # ── Build vectorised envs ─────────────────────────────────────────────────
    # SubprocVecEnv = true parallelism (one process per env)
    # Falls back to DummyVecEnv (sequential) on any import/spawn error.
    try:
        vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)],
                                start_method="spawn")
        print(f"[ENV] SubprocVecEnv with {n_envs} parallel environments")
    except Exception as exc:
        print(f"[WARN] SubprocVecEnv failed ({exc}), falling back to DummyVecEnv")
        vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    vec_env = VecFrameStack(vec_env, n_stack=4)     # 4-frame stack → velocity info
    vec_env = VecTransposeImage(vec_env)             # H×W×C → C×H×W (PyTorch)
    vec_env = VecMonitor(vec_env, log_dir)           # reward/length logging for TB

    # Separate eval env (single, deterministic)
    eval_env = DummyVecEnv([make_env(0, seed=9999)])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)               # match type with training env

    # ── PPO Hyperparameters ───────────────────────────────────────────────────
    # Tuned for CarRacing image-based continuous control.
    # Key references:
    #   • SB3 RL Zoo defaults for CarRacing
    #   • Schulman et al. (2017) PPO paper
    #   • Community benchmarks on Gymnasium CarRacing-v2/v3
    hyperparams = dict(
        policy         = "CnnPolicy",
        learning_rate  = 3e-4,       # Adam LR; 3e-4 is a reliable default
        n_steps        = 512,        # Steps per env per rollout; 512×4=2048 total
        batch_size     = 128,        # Mini-batch for gradient update
        n_epochs       = 10,         # Number of passes over each rollout buffer
        gamma          = 0.99,       # Discount factor
        gae_lambda     = 0.95,       # GAE smoothing (bias-variance trade-off)
        clip_range     = 0.2,        # PPO clip ε
        ent_coef       = 0.01,       # Entropy bonus (encourages exploration)
        vf_coef        = 0.5,        # Value-function loss coefficient
        max_grad_norm  = 0.5,        # Gradient clipping
        verbose        = 1,
        tensorboard_log= log_dir,
        device         = device,
    )

    model = PPO(env=vec_env, **hyperparams)

    print("\n──────────────────────────────────────")
    print(f"Policy network:\n{model.policy}")
    print("──────────────────────────────────────\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    # Checkpoint: save every 50 000 env steps (scaled for n_envs)
    if test_mode:
        total_timesteps = 5_000
        ckpt_freq       = max(1_000 // n_envs, 1)
    else:
        ckpt_freq       = max(50_000 // n_envs, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq  = ckpt_freq,
        save_path  = model_dir,
        name_prefix= "ppo_carracing",
        verbose    = 1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = model_dir,
        log_path             = log_dir,
        eval_freq            = max(50_000 // n_envs, 1),
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1,
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"Training for {total_timesteps:,} timesteps …")
    model.learn(
        total_timesteps   = total_timesteps,
        callback          = callbacks,
        tb_log_name       = "PPO_CarRacing",
        reset_num_timesteps= True,
        progress_bar      = True,
    )

    # Save final model
    final_path = os.path.join(model_dir, "ppo_carracing_final")
    model.save(final_path)
    print(f"\n[DONE] Final model saved → {final_path}.zip")

    vec_env.close()
    eval_env.close()


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO CarRacing-v3 trainer")
    parser.add_argument("--test",       action="store_true",
                        help="Run a quick 5 000-step test instead of full training")
    parser.add_argument("--timesteps",  type=int, default=1_500_000,
                        help="Total env timesteps (ignored in --test mode)")
    parser.add_argument("--n-envs",     type=int, default=4,
                        help="Number of parallel environments")
    args = parser.parse_args()

    train(
        total_timesteps = args.timesteps,
        n_envs          = args.n_envs,
        test_mode       = args.test,
    )
