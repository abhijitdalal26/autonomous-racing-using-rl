import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_tensorboard_metrics(log_dir="./logs", output_dir="./images"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Locate all event files
    event_files = glob.glob(os.path.join(log_dir, "**/*tfevents*"), recursive=True)
    if not event_files:
        print("No TensorBoard event files found.")
        return
    
    # Find the largest event file, assuming it's the 1.5M full training run
    main_event_file = max(event_files, key=os.path.getsize)
    print(f"Loading data from: {main_event_file}")
    
    # Load data
    ea = EventAccumulator(main_event_file, size_guidance={'scalars': 0})
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    
    metrics_to_plot = {
        'rollout/ep_rew_mean': 'Training Reward (Mean Episode Reward)',
        'train/value_loss': 'Value Loss',
        'train/policy_gradient_loss': 'Policy Gradient Loss',
        'train/entropy_loss': 'Entropy Loss',
        'train/learning_rate': 'Learning Rate'
    }
    
    for tag, title in metrics_to_plot.items():
        if tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            plt.figure(figsize=(10, 6))
            plt.plot(steps, values, color='#1f77b4', linewidth=2)
            plt.title(f'PPO CarRacing-v3: {title}', fontsize=14, fontweight='bold')
            plt.xlabel('Timesteps', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            safe_name = tag.replace("/", "_")
            out_file = os.path.join(output_dir, f"plot_{safe_name}.png")
            plt.savefig(out_file, dpi=300)
            plt.close()
            print(f"Saved {out_file}")

if __name__ == '__main__':
    plot_tensorboard_metrics()
