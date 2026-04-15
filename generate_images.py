import os
import cv2
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

def capture_frame(model_path, output_name, steps_to_run):
    if not os.path.exists(model_path):
        print(f'Skipping {model_path}, file not found')
        return
        
    def _init():
        env = gym.make('CarRacing-v3', continuous=True, render_mode='rgb_array')
        env = GrayscaleObservation(env, keep_dim=True)
        return env

    inf_env = DummyVecEnv([_init])
    inf_env = VecFrameStack(inf_env, n_stack=4)
    inf_env = VecTransposeImage(inf_env)
    
    raw_env = gym.make('CarRacing-v3', continuous=True, render_mode='rgb_array')

    model = PPO.load(model_path, device='auto')
    
    seed = 42
    inf_env.seed(seed)
    raw_env.unwrapped.reset(seed=seed)
    
    obs = inf_env.reset()
    raw_obs, _ = raw_env.reset(seed=seed)
    
    frame = None
    for _ in range(steps_to_run):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _ = inf_env.step(action)
        raw_action = action[0]
        _, _, raw_term, _, _ = raw_env.step(raw_action)
        frame = raw_env.render()
        if terminated[0] or raw_term:
            break
            
    if frame is not None:
        cv2.imwrite(output_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f'Saved {output_name}')

    inf_env.close()
    raw_env.close()

os.makedirs('images', exist_ok=True)
capture_frame('models/ppo_carracing_50000_steps.zip', 'images/01_early_model_off_road.jpg', 60)
capture_frame('models/ppo_carracing_400000_steps.zip', 'images/02_mid_model_cornering.jpg', 200)
capture_frame('models/best_model.zip', 'images/03_best_model_solved.jpg', 300)
