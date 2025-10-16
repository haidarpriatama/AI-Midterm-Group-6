import numpy as np
import torch
from rocket_env_with_reward import SimpleRocketEnv
from dqn_variants import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PrioritizedDQNAgent
import pygame
import cv2
import os

def load_agent(agent_type, model_path, state_dim, action_dim):
    """Load trained agent from file"""
    if agent_type == 'DQN':
        agent = DQNAgent(state_dim, action_dim)
    elif agent_type == 'Double DQN':
        agent = DoubleDQNAgent(state_dim, action_dim)
    elif agent_type == 'Dueling DQN':
        agent = DuelingDQNAgent(state_dim, action_dim)
    elif agent_type == 'Prioritized DQN':
        agent = PrioritizedDQNAgent(state_dim, action_dim)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()
    return agent


def test_agent_with_video(agent, env, episodes=10, video_path='output_video.mp4', fps=30):
    """Test agent and save video"""
    
    # Video writer setup
    frame_size = (env.screen_w, env.screen_h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    
    episode_rewards = []
    success_count = 0
    
    print(f"Recording {episodes} episodes to {video_path}...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        frame_count = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        while not (done or truncated):
            # Select action (greedy)
            action = agent.select_action(state, epsilon=0.0)
            
            # Take step
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            # Render and capture frame
            env.render()
            
            # Capture the pygame screen
            frame = pygame.surfarray.array3d(env.screen)
            frame = frame.transpose([1, 0, 2])  # Pygame uses (width, height, channels)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            video_writer.write(frame)
            
            frame_count += 1
        
        episode_rewards.append(episode_reward)
        if episode_reward > 500:
            success_count += 1
            print(f"  Result: SUCCESS! Reward: {episode_reward:.2f}")
        else:
            print(f"  Result: Failed. Reward: {episode_reward:.2f}")
    
    video_writer.release()
    
    print(f"\n{'='*60}")
    print(f"Video saved to: {video_path}")
    print(f"Total episodes: {episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Success rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return episode_rewards


def test_agent_interactive(agent, env):
    """Interactive testing - watch the agent perform"""
    print("\nInteractive Testing Mode")
    print("Press SPACE to start new episode, ESC to quit")
    
    running = True
    episode = 0
    
    while running:
        state, _ = env.reset()
        episode += 1
        episode_reward = 0
        done = False
        truncated = False
        
        print(f"\nEpisode {episode} - Starting...")
        
        while not (done or truncated) and running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Select action
            action = agent.select_action(state, epsilon=0.0)
            
            # Take step
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            # Render
            env.render()
        
        if episode_reward > 500:
            print(f"Episode {episode} - SUCCESS! Reward: {episode_reward:.2f}")
        else:
            print(f"Episode {episode} - Failed. Reward: {episode_reward:.2f}")
        
        if running:
            print("Press SPACE for another episode, or close window to exit...")


def compare_all_agents_video(video_path='all_agents_comparison.mp4', episodes_per_agent=3):
    """Create video comparing all agents"""
    
    state_dim = 9
    action_dim = 4
    
    agents_info = [
        ('DQN', 'models/dqn_model.pth'),
        ('Double DQN', 'models/double_dqn_model.pth'),
        ('Dueling DQN', 'models/dueling_dqn_model.pth'),
        ('Prioritized DQN', 'models/prioritized_dqn_model.pth')
    ]
    
    env = SimpleRocketEnv(render_mode='human')
    frame_size = (env.screen_w, env.screen_h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, frame_size)
    
    print(f"Creating comparison video: {video_path}")
    
    for agent_name, model_path in agents_info:
        if not os.path.exists(model_path):
            print(f"Warning: Model not found: {model_path}")
            continue
        
        print(f"\nRecording {agent_name}...")
        agent = load_agent(agent_name, model_path, state_dim, action_dim)
        
        for ep in range(episodes_per_agent):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            # Add title frame
            for _ in range(60):  # 2 seconds at 30fps
                env.render()
                frame = pygame.surfarray.array3d(env.screen)
                frame = frame.transpose([1, 0, 2])
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add text overlay
                text = f"{agent_name} - Episode {ep+1}"
                cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 0), 2, cv2.LINE_AA)
                
                video_writer.write(frame)
            
            while not (done or truncated):
                action = agent.select_action(state, epsilon=0.0)
                next_state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                state = next_state
                
                env.render()
                frame = pygame.surfarray.array3d(env.screen)
                frame = frame.transpose([1, 0, 2])
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add overlay
                cv2.putText(frame, f"{agent_name}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Reward: {episode_reward:.1f}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
                video_writer.write(frame)
            
            print(f"  Episode {ep+1}: Reward = {episode_reward:.2f}")
    
    video_writer.release()
    env.close()
    print(f"\nComparison video saved to: {video_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained DQN agents')
    parser.add_argument('--agent', type=str, default='Dueling DQN',
                       choices=['DQN', 'Double DQN', 'Dueling DQN', 'Prioritized DQN'],
                       help='Agent type to test')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (default: models/{agent}_model.pth)')
    parser.add_argument('--mode', type=str, default='video',
                       choices=['video', 'interactive', 'compare'],
                       help='Testing mode')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to record')
    parser.add_argument('--output', type=str, default='outputs/test_video.mp4',
                       help='Output video path')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    if args.mode == 'compare':
        compare_all_agents_video(args.output, episodes_per_agent=3)
    else:
        # Setup model path
        if args.model is None:
            model_name = args.agent.replace(' ', '_').lower()
            args.model = f'models/{model_name}_model.pth'
        
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            print("Please train the model first using train_and_compare.py")
            return
        
        # Load agent
        env = SimpleRocketEnv(render_mode='human')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = load_agent(args.agent, args.model, state_dim, action_dim)
        print(f"Loaded {args.agent} from {args.model}")
        
        if args.mode == 'video':
            test_agent_with_video(agent, env, episodes=args.episodes, video_path=args.output)
        elif args.mode == 'interactive':
            test_agent_interactive(agent, env)
        
        env.close()


if __name__ == "__main__":
    main()