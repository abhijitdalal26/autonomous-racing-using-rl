"""
infer.py — Evaluate a saved PPO checkpoint on CarRacing-v3
===========================================================
Loads any saved .zip model, runs one full episode, prints the
total reward, and saves an MP4 video that you can embed in your report.

Usage:
  # Evaluate a specific checkpoint (renders to screen + saves video)
  conda run -n car-rl python infer.py --model models/ppo_carracing_100_steps.zip

  # Evaluate the best model saved by EvalCallback
  conda run -n car-rl python infer.py --model models/best_model.zip

  # Save video only (no on-screen window) — useful for headless servers
  conda run -n car-rl python infer.py --model models/best_model.zip --no-render
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_eval_env(render: bool = False):
    """Build a single wrapped CarRacing-v3 env for evaluation."""
    render_mode = "rgb_array"   # always capture pixels for video

    def _init():
        env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
        env = GrayscaleObservation(env, keep_dim=True)
        return env

    vec_env = DummyVecEnv([_init])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)
    return vec_env


def save_video(frames: list, output_path: str, fps: int = 30):
    """Write a list of RGB frames to an MP4 file using OpenCV."""
    if not frames:
        print("[WARN] No frames recorded, skipping video.")
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"[VIDEO] Saved → {output_path}  ({len(frames)} frames @ {fps} fps)")


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model_path: str, render: bool = True, video_out: str = ""):
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[LOAD] {model_path}")
    # We set device="auto" so it picks up CUDA if available
    model = PPO.load(model_path, device="auto")

    # ── a raw (un-stacked) env only for pixel capture ─────────────────────
    raw_env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")

    # ── the stacked env for model inference ──────────────────────────────
    inf_env = make_eval_env(render=render)

    obs = inf_env.reset()
    raw_obs, _ = raw_env.reset()

    total_reward = 0.0
    frames       = []
    step         = 0

    print("[RUN] Episode started …")

    # Window for real-time display
    if render:
        cv2.namedWindow("CarRacing-v3", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CarRacing-v3", 600, 600)

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = inf_env.step(action)

        # Sync the raw env to capture full-colour frames
        raw_action = action[0]   # un-batch
        raw_obs, raw_rew, raw_term, raw_trunc, _ = raw_env.step(raw_action)
        frame = raw_env.render()
        frames.append(frame)

        total_reward += reward[0]
        step         += 1

        # Live display
        if render and frame is not None:
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("CarRacing-v3", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[STOP] User pressed Q")
                break

        if terminated[0] or (info[0].get("TimeLimit.truncated", False)):
            print(f"\n[DONE] Episode finished — steps={step}  total_reward={total_reward:.2f}")
            break

    if render:
        cv2.destroyAllWindows()

    inf_env.close()
    raw_env.close()

    # ── Save video ────────────────────────────────────────────────────────
    if not video_out:
        base = os.path.splitext(os.path.basename(model_path))[0]
        os.makedirs("./videos", exist_ok=True)
        video_out = f"./videos/{base}.mp4"

    save_video(frames, video_out)
    return total_reward


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PPO CarRacing checkpoint")
    parser.add_argument("--model",     required=True,
                        help="Path to .zip model file")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable on-screen window (video is still saved)")
    parser.add_argument("--video-out", default="",
                        help="Custom output path for the MP4 (optional)")
    args = parser.parse_args()

    reward = evaluate(
        model_path = args.model,
        render     = not args.no_render,
        video_out  = args.video_out,
    )
    print(f"[RESULT] Total reward: {reward:.2f}")
