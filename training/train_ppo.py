#!/usr/bin/env python3
"""
ANTIGRAVITY — PPO Navigation Policy Training Script
=====================================================
Trains a PPO policy using Stable-Baselines3 on the DroneNavEnv.

Usage:
  python training/train_ppo.py --timesteps 500000 --output models/ppo_navigation
  python training/train_ppo.py --algo sac --timesteps 1000000
"""

import argparse
import os
import sys
import time

# Add training directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def make_env(seed=0):
    """Create and seed the drone nav environment."""
    from drone_nav_env import DroneNavEnv
    env = DroneNavEnv(map_size=30.0, max_steps=500)
    env.reset(seed=seed)
    return env


def train(args):
    try:
        from stable_baselines3 import PPO, SAC, A2C
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import (
            EvalCallback, CheckpointCallback, CallbackList)
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("Error: stable-baselines3 not installed.")
        print("  pip install stable-baselines3[extra]")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  ANTIGRAVITY — PPO Navigation Training")
    print(f"  Algorithm: {args.algo.upper()}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"{'='*60}")

    # Create vectorized environments
    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=42)
    eval_env = Monitor(make_env(seed=99))

    # Algorithm selection
    algo_map = {'ppo': PPO, 'sac': SAC, 'a2c': A2C}
    AlgoClass = algo_map.get(args.algo.lower(), PPO)

    # Hyperparameters
    if args.algo.lower() == 'ppo':
        model = AlgoClass(
            'MlpPolicy', vec_env, verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            tensorboard_log=os.path.join(args.output, 'tb_logs'),
        )
    elif args.algo.lower() == 'sac':
        model = AlgoClass(
            'MlpPolicy', vec_env, verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=os.path.join(args.output, 'tb_logs'),
        )
    else:
        model = AlgoClass('MlpPolicy', vec_env, verbose=1)

    # Callbacks
    os.makedirs(args.output, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.timesteps // 10, 1000),
        save_path=os.path.join(args.output, 'checkpoints'),
        name_prefix='antigravity')
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=os.path.join(args.output, 'best'),
        log_path=os.path.join(args.output, 'eval_logs'),
        eval_freq=max(args.timesteps // 20, 500),
        n_eval_episodes=10, deterministic=True)
    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # Train
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
    train_time = time.time() - t0

    # Save final model
    final_path = os.path.join(args.output, f'{args.algo}_navigation')
    model.save(final_path)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Time: {train_time:.0f}s ({train_time/3600:.1f}h)")
    print(f"  Model saved: {final_path}.zip")
    print(f"  Best model: {args.output}/best/best_model.zip")
    print(f"{'='*60}")

    # Quick evaluation
    print("\nRunning evaluation...")
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ANTIGRAVITY navigation policy')
    parser.add_argument('--algo', default='ppo', choices=['ppo', 'sac', 'a2c'])
    parser.add_argument('--timesteps', type=int, default=500000)
    parser.add_argument('--n-envs', type=int, default=4)
    parser.add_argument('--output', default='models/rl_training')
    args = parser.parse_args()
    train(args)
