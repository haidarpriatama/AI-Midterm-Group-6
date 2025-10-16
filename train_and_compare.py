import numpy as np
import matplotlib.pyplot as plt
from rocket_env_with_reward import SimpleRocketEnv
from dqn_variants import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PrioritizedDQNAgent
import torch
from tqdm import tqdm
import json
import os

def train_agent(agent, env, episodes=2000, epsilon_start=1.0, epsilon_end=0.01, 
                epsilon_decay=0.995, save_path=None):
    """Train a DQN agent"""
    epsilon = epsilon_start
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    print(f"Training {agent.__class__.__name__}...")
    
    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated):
            # Select action
            action = agent.select_action(state, epsilon)
            
            # Take step
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, done or truncated)
            
            # Update agent
            loss = agent.update()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"\nEpisode {episode+1}/{episodes}")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Avg Length (last 100): {avg_length:.2f}")
            print(f"  Epsilon: {epsilon:.4f}")
    
    # Save model
    if save_path:
        torch.save(agent.policy_net.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'losses': losses
    }


def evaluate_agent(agent, env, episodes=50, render=False):
    """Evaluate trained agent"""
    episode_rewards = []
    success_count = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Always greedy during evaluation
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, done, truncated, _ = env.step(action)
            
            if render:
                env.render()
            
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        # Check if landing was successful (high reward)
        if episode_reward > 500:
            success_count += 1
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': success_count / episodes,
        'all_rewards': episode_rewards
    }


def plot_training_results(results_dict, save_path='training_comparison.png'):
    """Plot training results for multiple agents"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    for name, results in results_dict.items():
        rewards = results['rewards']
        # Smooth with moving average
        window = 50
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=name, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Rewards (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Average Reward per 100 episodes
    ax = axes[0, 1]
    for name, results in results_dict.items():
        rewards = results['rewards']
        avg_rewards = [np.mean(rewards[max(0, i-100):i+1]) 
                      for i in range(len(rewards))]
        ax.plot(avg_rewards, label=name, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward (last 100 episodes)')
    ax.set_title('Average Reward Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Episode Lengths
    ax = axes[1, 0]
    for name, results in results_dict.items():
        lengths = results['lengths']
        window = 50
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=name, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Lengths (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Loss
    ax = axes[1, 1]
    for name, results in results_dict.items():
        if results['losses']:
            losses = results['losses']
            window = 50
            if len(losses) > window:
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax.plot(smoothed, label=name, alpha=0.8)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training comparison plot saved to {save_path}")
    plt.close()


def plot_evaluation_results(eval_results, save_path='evaluation_comparison.png'):
    """Plot evaluation results comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bar chart of mean rewards
    ax = axes[0]
    names = list(eval_results.keys())
    means = [eval_results[name]['mean_reward'] for name in names]
    stds = [eval_results[name]['std_reward'] for name in names]
    
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(names)])
    ax.set_ylabel('Mean Reward')
    ax.set_title('Evaluation: Mean Reward ± Std')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Success rate
    ax = axes[1]
    success_rates = [eval_results[name]['success_rate'] * 100 for name in names]
    bars = ax.bar(x_pos, success_rates, alpha=0.7,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(names)])
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Evaluation: Landing Success Rate')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation comparison plot saved to {save_path}")
    plt.close()


def main():
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Training environment (no rendering for speed)
    train_env = SimpleRocketEnv(render_mode=None)
    
    # Hyperparameters
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    
    config = {
        'lr': 3e-4,              # Slightly lower learning rate
        'gamma': 0.99,
        'buffer_size': 50000,
        'batch_size': 128,
        'target_update': 500,
        'episodes': 2000,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,     # Changed from 0.01 (more exploration)
        'epsilon_decay': 0.998   # Changed from 0.995 (slower decay)
    }
    
    # Initialize agents
    agents = {
        'DQN': DQNAgent(state_dim, action_dim, lr=config['lr'], 
                       gamma=config['gamma'], buffer_size=config['buffer_size'],
                       batch_size=config['batch_size'], target_update=config['target_update']),
        'Double DQN': DoubleDQNAgent(state_dim, action_dim, lr=config['lr'],
                                     gamma=config['gamma'], buffer_size=config['buffer_size'],
                                     batch_size=config['batch_size'], target_update=config['target_update']),
        'Dueling DQN': DuelingDQNAgent(state_dim, action_dim, lr=config['lr'],
                                       gamma=config['gamma'], buffer_size=config['buffer_size'],
                                       batch_size=config['batch_size'], target_update=config['target_update']),
        'Prioritized DQN': PrioritizedDQNAgent(state_dim, action_dim, lr=config['lr'],
                                               gamma=config['gamma'], buffer_size=config['buffer_size'],
                                               batch_size=config['batch_size'], target_update=config['target_update'])
    }
    
    # Train all agents
    training_results = {}
    for name, agent in agents.items():
        model_path = f'models/{name.replace(" ", "_").lower()}_model.pth'
        results = train_agent(
            agent, train_env,
            episodes=config['episodes'],
            epsilon_start=config['epsilon_start'],
            epsilon_end=config['epsilon_end'],
            epsilon_decay=config['epsilon_decay'],
            save_path=model_path
        )
        training_results[name] = results
        print(f"\n{name} training completed!\n")
    
    # Plot training results
    plot_training_results(training_results, 'outputs/training_comparison.png')
    
    # Evaluate all agents
    eval_env = SimpleRocketEnv(render_mode=None)
    eval_results = {}
    
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    
    for name, agent in agents.items():
        print(f"\nEvaluating {name}...")
        results = evaluate_agent(agent, eval_env, episodes=100)
        eval_results[name] = results
        
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Success Rate: {results['success_rate']*100:.1f}%")
    
    # Plot evaluation results
    plot_evaluation_results(eval_results, 'outputs/evaluation_comparison.png')
    
    # Save results to JSON
    summary = {
        'config': config,
        'evaluation': {
            name: {
                'mean_reward': float(results['mean_reward']),
                'std_reward': float(results['std_reward']),
                'success_rate': float(results['success_rate'])
            }
            for name, results in eval_results.items()
        }
    }
    
    with open('outputs/results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION COMPLETED!")
    print("="*60)
    print("\nResults saved to 'outputs/' directory")
    print("Models saved to 'models/' directory")
    
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()