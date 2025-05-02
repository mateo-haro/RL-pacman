import gymnasium as gym
import ale_py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import yaml
import psutil
import os
import torch
from preprocessing import create_env, get_state
from agent import DQNAgent


def load_hyperparameters(config_path='hyperparameters.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        gpu_mem_max = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
        return cpu_mem, gpu_mem, gpu_mem_max
    else:
        return cpu_mem, 0, 0


def train(episodes, agent=None, env=None, frame_skip=4):
    scores = []
    cpu_memory_usage = []
    gpu_memory_usage = []
    gpu_memory_max = []
    avg_rewards = []  # Track average rewards
    avg_losses = []   # Track average losses
    
    for episode in tqdm(range(episodes)):
        obs, info = env.reset()
        score = 0
        total_reward = 0
        nb_lives = 3
        episode_over = False
        
        # Record memory usage at the start of each episode
        cpu_mem, gpu_mem, gpu_max = get_memory_usage()
        cpu_memory_usage.append(cpu_mem)
        gpu_memory_usage.append(gpu_mem)
        gpu_memory_max.append(gpu_max)
        
        while not episode_over:
            action = agent.act(obs)
            stacked_reward = 0

            # Perform frame skipping
            for _ in range(frame_skip):
                next_obs, reward, terminated, truncated, info = env.step(action)
                score += reward

                #Engineered reward for better training

                # penalize death and reward staying alive
                training_reward = 0
                if info["lives"] < nb_lives:
                    nb_lives = info["lives"]
                    training_reward = -50
                elif info["lives"] == nb_lives:
                    training_reward += 0.01

                # reward for collecting coins
                if reward > 0:
                    training_reward += 1
                episode_over = terminated or truncated
                if episode_over:
                    break

                stacked_reward += training_reward

            agent.remember(obs, action, stacked_reward, next_obs, terminated)
            obs = next_obs
            agent.replay_training()
            total_reward += stacked_reward

        scores.append(score)

        # Print progress and memory usage
        if episode < 10:
            print(f"Episode {episode + 1}, Score: {total_reward}, CPU Memory: {cpu_memory_usage[-1]:.2f} MB, GPU Memory: {gpu_memory_usage[-1]:.2f} MB, GPU Max: {gpu_memory_max[-1]:.2f} MB")
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            avg_cpu_mem = np.mean(cpu_memory_usage[-10:])
            avg_gpu_mem = np.mean(gpu_memory_usage[-10:])
            max_gpu_mem = np.max(gpu_memory_max[-10:])
            
            # Calculate average loss for the last 10 episodes
            if len(agent.losses) > 0:
                avg_loss = np.mean(agent.losses[-100:])  # Average over last 100 losses
                avg_losses.append(avg_loss)
                agent.losses = []  # Clear losses for next batch
            else:
                avg_loss = 0
                avg_losses.append(0)
            
            avg_rewards.append(avg_score)
            
            print(f"Episode {episode + 1}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}, Average Loss: {avg_loss:.4f}")
            print(f"Average CPU Memory: {avg_cpu_mem:.2f} MB, Average GPU Memory: {avg_gpu_mem:.2f} MB, Max GPU Memory: {max_gpu_mem:.2f} MB")
            
            # Save intermediate model
            agent.save(f'models/dqn_agent_{episode + 1}.pth')
            
            # Plot scores, memory usage, and training metrics
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot scores
            ax1.plot(scores)
            ax1.set_title('Training Progress')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Score')
            
            # Plot average rewards
            ax2.plot(avg_rewards)
            ax2.set_title('Average Rewards (Every 10 Episodes)')
            ax2.set_xlabel('Episode (Every 10)')
            ax2.set_ylabel('Average Reward')
            
            # Plot memory usage
            ax3.plot(cpu_memory_usage, label='CPU')
            ax3.plot(gpu_memory_usage, label='GPU Current')
            ax3.plot(gpu_memory_max, label='GPU Max')
            ax3.set_title('Memory Usage')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Memory (MB)')
            ax3.legend()
            
            # Plot average loss
            ax4.plot(avg_losses)
            ax4.set_title('Average Loss (Every 10 Steps)')
            ax4.set_xlabel('Episode (Every 10)')
            ax4.set_ylabel('Average Loss')
            
            plt.tight_layout()
            plt.savefig('training_progress.png')
            plt.close()

    # Save final model
    agent.save('models/final_model.pth')

    env.close()

def test_policy(model_path, agent=None, env=None, frame_skip=4):
    """Load a trained model and run it in the environment with human rendering."""
    agent.load(model_path)
    
    # Run episodes with loaded model
    obs, info = env.reset()
    total_reward = 0
    episode_over = False
    
    while not episode_over:
        action = agent.act(obs)
        for _ in range(frame_skip):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_over = terminated or truncated
            if episode_over:
                break
        
    print(f"Episode finished with reward: {total_reward}")


def main():

    # Load hyperparameters
    config = load_hyperparameters()
    
    gym.register_envs(ale_py)

    # Create environment with config
    def create_env(config):
        env = create_env(
            render_mode=config['render_mode'],
            stack_size=config['stack_size'],
            resize_shape=config['resize_shape']
        )
        return env
    
    envs = gym.vector.AsyncVectorEnv([create_env(config['environment']) for _ in range(8)])


    # Create agent with config
    agent = DQNAgent(
        envs.observation_space.shape,
        envs.action_space.n,
        memory_size=config['agent']['memory_size'],
        batch_size=config['agent']['batch_size'],
        gamma=config['agent']['gamma'],
        epsilon=config['agent']['epsilon'],
        epsilon_min=config['agent']['epsilon_min'],
        epsilon_decay=config['agent']['epsilon_decay'],
        learning_rate=config['agent']['learning_rate']
    )

    if config['training']['train_flag']:
        train(episodes=config['training']['episodes'], agent=agent, env=env, frame_skip=config['environment']['stack_size'])
    else:
        test_policy(model_path='models/final_model.pth', agent=agent, env=env, frame_skip=config['environment']['stack_size'])

if __name__ == "__main__":
    main()