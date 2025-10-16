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


def add_info_overlay(frame, agent_name, episode, episode_reward, mean_reward, is_success):
    """Add information overlay to frame in top-right corner"""
    # Create semi-transparent background
    overlay = frame.copy()
    
    # Position for text (top-right corner)
    x_pos = frame.shape[1] - 280
    y_start = 10
    
    # Background rectangle
    cv2.rectangle(overlay, (x_pos - 10, y_start), 
                  (frame.shape[1] - 10, y_start + 130), 
                  (0, 0, 0), -1)
    
    # Blend overlay
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    small_font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    
    # Add text lines
    y = y_start + 25
    cv2.putText(frame, f"Variant: {agent_name}", (x_pos, y), 
                font, small_font_scale, (100, 200, 255), thickness, cv2.LINE_AA)
    
    y += 25
    cv2.putText(frame, f"Episode: {episode}", (x_pos, y), 
                font, small_font_scale, color, thickness, cv2.LINE_AA)
    
    y += 25
    cv2.putText(frame, f"Reward: {episode_reward:.1f}", (x_pos, y), 
                font, small_font_scale, color, thickness, cv2.LINE_AA)
    
    y += 25
    cv2.putText(frame, f"Mean: {mean_reward:.1f}", (x_pos, y), 
                font, small_font_scale, color, thickness, cv2.LINE_AA)
    
    y += 25
    status_color = (0, 255, 0) if is_success else (0, 0, 255)
    status_text = "SUCCESS" if is_success else "FAILED"
    cv2.putText(frame, f"Status: {status_text}", (x_pos, y), 
                font, small_font_scale, status_color, thickness + 1, cv2.LINE_AA)
    
    return frame


def test_all_agents_with_video(video_path='outputs/all_variants_comparison.mp4', episodes=8):
    """Test all 4 variants with 8 episodes each and save to single video"""
    
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
    print(f"Testing {len(agents_info)} variants with {episodes} episodes each\n")
    
    for agent_idx, (agent_name, model_path) in enumerate(agents_info):
        if not os.path.exists(model_path):
            print(f"Warning: Model not found: {model_path}")
            continue
        
        print(f"[{agent_idx + 1}/{len(agents_info)}] Recording {agent_name}...")
        agent = load_agent(agent_name, model_path, state_dim, action_dim)
        
        episode_rewards = []
        
        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                action = agent.select_action(state, epsilon=0.0)
                next_state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                state = next_state
                
                env.render()
                
                # Capture frame
                frame = pygame.surfarray.array3d(env.screen)
                frame = frame.transpose([1, 0, 2])
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Calculate current mean
                current_mean = np.mean(episode_rewards + [episode_reward]) if episode_rewards else episode_reward
                is_success = episode_reward > 500
                
                # Add overlay
                frame = add_info_overlay(frame, agent_name, ep + 1, episode_reward, 
                                        current_mean, is_success)
                
                video_writer.write(frame)
            
            episode_rewards.append(episode_reward)
            success = "✓" if episode_reward > 500 else "✗"
            print(f"  Episode {ep+1}/{episodes}: Reward = {episode_reward:.2f} {success}")
        
        # Print summary for this agent
        mean_reward = np.mean(episode_rewards)
        success_count = sum(1 for r in episode_rewards if r > 500)
        print(f"  Summary: Mean = {mean_reward:.2f}, Success = {success_count}/{episodes}\n")
    
    video_writer.release()
    env.close()
    print(f"{'='*60}")
    print(f"Video saved to: {video_path}")
    print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained DQN agents')
    parser.add_argument('--episodes', type=int, default=8,
                       help='Number of episodes per variant')
    parser.add_argument('--output', type=str, default='outputs/all_variants_comparison.mp4',
                       help='Output video path')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run the test
    test_all_agents_with_video(video_path=args.output, episodes=args.episodes)


if __name__ == "__main__":
    main()
