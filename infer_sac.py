"""
infer_sac.py — Evaluate a saved SAC checkpoint on CarRacing-v3
==============================================================
Loads any saved .zip model, runs one full episode, prints the
total reward, and saves an MP4 video that you can embed in your report.

Usage:
  # Evaluate a specific checkpoint
  conda run -n car-rl python infer_sac.py --model models/sac_carracing_100000_steps.zip

  # Evaluate the best model saved by EvalCallback
  conda run -n car-rl python infer_sac.py --model models/best_model_sac.zip
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
import cv2


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


def evaluate(model_path: str, render: bool = True, video_out: str = ""):
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[LOAD] Loading SAC model from: {model_path}")
    model = SAC.load(model_path, device="auto")

    # Raw environment for pixel capture
    raw_env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    inf_env = make_eval_env(render=render)

    seed = 42
    inf_env.seed(seed)
    raw_env.unwrapped.reset(seed=seed)

    obs = inf_env.reset()
    raw_obs, _ = raw_env.reset(seed=seed)

    total_reward = 0.0
    frames       = []
    step         = 0

    print("[RUN] Episode started …")

    if render:
        cv2.namedWindow("CarRacing-v3 (SAC)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CarRacing-v3 (SAC)", 600, 600)

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = inf_env.step(action)

        raw_action = action[0]   
        raw_obs, raw_rew, raw_term, raw_trunc, _ = raw_env.step(raw_action)
        frame = raw_env.render()
        frames.append(frame)

        total_reward += reward[0]
        step         += 1

        if render and frame is not None:
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("CarRacing-v3 (SAC)", display)
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

    if not video_out:
        base = os.path.splitext(os.path.basename(model_path))[0]
        os.makedirs("./videos", exist_ok=True)
        video_out = f"./videos/{base}.mp4"

    save_video(frames, video_out)
    return total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a SAC CarRacing checkpoint")
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
