import argparse
import gymnasium as gym
import ale_py
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import yaml
import psutil
import os
import torch
from preprocessing import create_env, get_state, create_envs
from agent import DQNAgent
import wandb

MA_WINDOW = 200  # moving-average window for episode-based metrics


def load_hyperparameters(config_path='hyperparameters.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 / 1024
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_mem_max = torch.cuda.max_memory_allocated() / 1024 / 1024
        return cpu_mem, gpu_mem, gpu_mem_max
    return cpu_mem, 0, 0


def run_greedy_eval(agent, env, frame_skip=4, num_episodes=10):
    """Run evaluation episodes with a pure greedy policy (epsilon=0)."""
    eval_scores = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        score = 0.0
        done = False
        while not done:
            action = agent.greedy_action(obs)
            for _ in range(frame_skip):
                obs, reward, terminated, truncated, _ = env.step(action)
                score += reward
                done = terminated or truncated
                if done:
                    break
        eval_scores.append(score)
    return float(np.mean(eval_scores)), float(np.std(eval_scores))


def save_training_plots(history, path='training_progress.png'):
    """Save a 3x2 grid of diagnostic plots to disk."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # 1) Score + moving average
    ax = axes[0, 0]
    ax.plot(history["scores"], alpha=0.3, label="score")
    if len(history["scores"]) >= MA_WINDOW:
        ma = np.convolve(history["scores"],
                         np.ones(MA_WINDOW) / MA_WINDOW, mode='valid')
        ax.plot(range(MA_WINDOW - 1, len(history["scores"])), ma,
                label=f"MA{MA_WINDOW}")
    ax.set_title("Episode Score")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.legend()

    # 2) Episode length
    ax = axes[0, 1]
    ax.plot(history["episode_lengths"], alpha=0.3, label="length")
    if len(history["episode_lengths"]) >= MA_WINDOW:
        ma = np.convolve(history["episode_lengths"],
                         np.ones(MA_WINDOW) / MA_WINDOW, mode='valid')
        ax.plot(range(MA_WINDOW - 1, len(history["episode_lengths"])), ma,
                label=f"MA{MA_WINDOW}")
    ax.set_title("Episode Length (decision steps)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend()

    # 3) Training loss
    ax = axes[1, 0]
    ax.plot(history["losses"], alpha=0.4)
    ax.set_title("Training Loss (per step)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Huber loss")

    # 4) Mean TD error
    ax = axes[1, 1]
    ax.plot(history["td_errors"], alpha=0.4)
    ax.set_title("Mean |TD Error| (per step)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("|TD error|")

    # 5) Q-value statistics
    ax = axes[2, 0]
    ax.plot(history["mean_qs"], alpha=0.5, label="mean Q")
    ax.plot(history["max_qs"], alpha=0.5, label="max Q")
    ax.set_title("Predicted Q-Values (per step)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Q-value")
    ax.legend()

    # 6) Greedy evaluation
    ax = axes[2, 1]
    if history["eval_episodes"] and history["eval_scores"]:
        ax.plot(history["eval_episodes"], history["eval_scores"], marker='o')
    ax.set_title("Greedy Evaluation Score")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Eval score")

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def train(episodes, agent=None, envs=None, frame_skip=4,
          avg_param=50, eval_interval=100, eval_episodes=10,
          train_steps_per_sim_step=4, shaping_alpha=0.01,
          shaping_gamma=0.99, reward_scale=1.0,
          life_loss_penalty_scale=1.0, config=None):
    wandb_config = (config or {}).get("wandb", {})
    wandb.init(
        project=wandb_config.get("project"),
        entity=wandb_config.get("entity"),
    )

    # ---- history for plots ----
    history = {
        "scores": [], "episode_lengths": [],
        "losses": [], "td_errors": [], "mean_qs": [], "max_qs": [],
        "eval_episodes": [], "eval_scores": [],
    }

    # Rolling windows for recent training-step metrics (logged per episode)
    recent_losses = deque(maxlen=MA_WINDOW)
    recent_td_errors = deque(maxlen=MA_WINDOW)
    recent_mean_qs = deque(maxlen=MA_WINDOW)
    recent_max_qs = deque(maxlen=MA_WINDOW)
    recent_grad_norms = deque(maxlen=MA_WINDOW)
    recent_grad_norms_pre_clip = deque(maxlen=MA_WINDOW)
    recent_grad_norms_post_clip = deque(maxlen=MA_WINDOW)
    recent_mean_is_weights = deque(maxlen=MA_WINDOW)
    recent_min_is_weights = deque(maxlen=MA_WINDOW)
    recent_max_is_weights = deque(maxlen=MA_WINDOW)

    # ---- per-env tracking ----
    obs, info = envs.reset()
    env_scores = np.zeros(envs.num_envs)
    env_lives = np.full(envs.num_envs, 3)
    env_steps = np.zeros(envs.num_envs, dtype=np.int64)
    total_pallets = np.full(envs.num_envs, 244.0)
    completed_episodes = 0

    # ---- evaluation env (single, synchronous) ----
    eval_env = create_env(
        render_mode=None,
        stack_size=config['environment']['stack_size'],
        resize_shape=config['environment']['resize_shape'])

    pbar = tqdm(total=episodes)

    while completed_episodes < episodes:
        env_steps += 1
        actions = agent.act(obs)
        stacked_rewards = np.zeros(envs.num_envs)
        episode_dones = np.zeros(envs.num_envs, dtype=bool)
        final_obs = [None] * envs.num_envs

        # Frame skipping: repeat the same action, accumulate rewards
        for _ in range(frame_skip):
            next_obs, rewards, terminated, truncated, infos = envs.step(
                actions)
            step_dones = terminated | truncated
            active_mask = ~episode_dones
            if not np.any(active_mask):
                continue

            lives = np.asarray(infos["lives"])
            rewards_active = rewards[active_mask]

            env_scores[active_mask] += rewards_active
            stacked_rewards[active_mask] += reward_scale * rewards_active
            life_lost_mask = active_mask & (lives < env_lives)
            if np.any(life_lost_mask):
                env_lives[life_lost_mask] = lives[life_lost_mask]
                stacked_rewards[life_lost_mask] += (
                    life_loss_penalty_scale * -1)

            pellet_mask = active_mask & ((rewards == 10) | (rewards == 50))
            if np.any(pellet_mask):
                previous_pellets = total_pallets[pellet_mask].copy()
                total_pallets[pellet_mask] -= 1

                # Potential-based shaping: F = γ·Φ(s') − Φ(s)
                # Φ(s) = −α · pellets_remaining
                phi_s = -shaping_alpha * previous_pellets
                phi_next = -shaping_alpha * total_pallets[pellet_mask]
                stacked_rewards[pellet_mask] += shaping_gamma * phi_next - phi_s

                depleted_mask = pellet_mask & (total_pallets == 0)
                if np.any(depleted_mask):
                    total_pallets[depleted_mask] = 244

            new_done_mask = active_mask & step_dones
            if np.any(new_done_mask):
                episode_dones[new_done_mask] = True
                if "_final_observation" in infos:
                    final_flags = np.asarray(infos["_final_observation"], dtype=bool)
                    copy_final_mask = new_done_mask & final_flags
                    if np.any(copy_final_mask):
                        final_obs_arr = infos["final_observation"]
                        for idx in np.flatnonzero(copy_final_mask):
                            final_obs[idx] = final_obs_arr[idx]

        store_next_obs = next_obs.copy()
        done_indices = np.flatnonzero(episode_dones)
        if done_indices.size > 0:
            total_pallets[done_indices] = 244
            for i in done_indices:
                if final_obs[i] is not None:
                    store_next_obs[i] = final_obs[i]

        agent.remember(obs, actions, stacked_rewards, store_next_obs,
                       episode_dones.astype(np.float32))
        obs = next_obs

        # ---- train & collect step-level metrics ----
        for _ in range(train_steps_per_sim_step):
            metrics = agent.replay_training()
        if metrics is not None:
            recent_losses.append(metrics["loss"])
            recent_td_errors.append(metrics["mean_td_error"])
            recent_mean_qs.append(metrics["mean_q"])
            recent_max_qs.append(metrics["max_q"])
            if "grad_norm" in metrics:
                recent_grad_norms.append(metrics["grad_norm"])
            if "grad_norm_pre_clip" in metrics:
                recent_grad_norms_pre_clip.append(metrics["grad_norm_pre_clip"])
            if "grad_norm_post_clip" in metrics:
                recent_grad_norms_post_clip.append(
                    metrics["grad_norm_post_clip"])
            if "mean_is_weight" in metrics:
                recent_mean_is_weights.append(metrics["mean_is_weight"])
                recent_min_is_weights.append(metrics["min_is_weight"])
                recent_max_is_weights.append(metrics["max_is_weight"])
            history["losses"].append(metrics["loss"])
            history["td_errors"].append(metrics["mean_td_error"])
            history["mean_qs"].append(metrics["mean_q"])
            history["max_qs"].append(metrics["max_q"])

        # ---- log completed episodes ----
        for i in done_indices:

            completed_episodes += 1
            ep_score = env_scores[i]
            ep_length = int(env_steps[i])
            history["scores"].append(ep_score)
            history["episode_lengths"].append(ep_length)
            pbar.update(1)

            # Moving average (use whatever we have so far, up to MA_WINDOW)
            ma_scores = np.mean(
                history["scores"][-MA_WINDOW:])

            # Wandb: episode-level
            log_dict = {
                "episode/score": ep_score,
                "episode/length": ep_length,
                "episode/epsilon": agent.epsilon,
                "episode/score_ma200": ma_scores,
                "episode/replay_size": len(agent.memory),
            }
            # Wandb: recent training-step averages
            if recent_losses:
                log_dict["train/loss"] = np.mean(recent_losses)
                log_dict["train/mean_td_error"] = np.mean(recent_td_errors)
                log_dict["train/mean_q"] = np.mean(recent_mean_qs)
                log_dict["train/max_q"] = max(recent_max_qs)
                if recent_grad_norms:
                    log_dict["train/grad_norm"] = np.mean(recent_grad_norms)
                if recent_grad_norms_pre_clip:
                    log_dict["train/grad_norm_pre_clip"] = np.mean(
                        recent_grad_norms_pre_clip)
                if recent_grad_norms_post_clip:
                    log_dict["train/grad_norm_post_clip"] = np.mean(
                        recent_grad_norms_post_clip)
                if recent_mean_is_weights:
                    log_dict["train/mean_is_weight"] = np.mean(
                        recent_mean_is_weights)
                    log_dict["train/min_is_weight"] = min(recent_min_is_weights)
                    log_dict["train/max_is_weight"] = max(recent_max_is_weights)

            wandb.log(log_dict, step=completed_episodes)

            # Console output for first 10 episodes
            if completed_episodes <= 10:
                print(f"\nEp {completed_episodes}  "
                      f"score={ep_score:.1f}  len={ep_length}  "
                      f"eps={agent.epsilon:.4f}")

            # Reset per-env tracking
            env_scores[i] = 0
            env_lives[i] = 3
            env_steps[i] = 0

            # ---- periodic console summary ----
            if completed_episodes % avg_param == 0:
                avg_s = np.mean(history["scores"][-avg_param:])
                cpu_m, gpu_m, gpu_mx = get_memory_usage()
                loss_str = (f"{np.mean(recent_losses):.4f}"
                            if recent_losses else "n/a")
                print(f"\n[{completed_episodes}/{episodes}] "
                      f"avg_score={avg_s:.1f}  eps={agent.epsilon:.4f}  "
                      f"loss={loss_str}  "
                      f"cpu={cpu_m:.0f}MB  gpu={gpu_m:.0f}MB")
                agent.save(
                    f'models/dqn_agent_{completed_episodes}.pth')

            # ---- greedy evaluation ----
            if completed_episodes % eval_interval == 0:
                eval_mean, eval_std = run_greedy_eval(
                    agent, eval_env, frame_skip, eval_episodes)
                history["eval_episodes"].append(completed_episodes)
                history["eval_scores"].append(eval_mean)
                wandb.log({
                    "eval/mean_score": eval_mean,
                    "eval/std_score": eval_std,
                }, step=completed_episodes)
                print(f"  [eval] mean={eval_mean:.1f}  std={eval_std:.1f}")

            # ---- save plots periodically ----
            if completed_episodes % avg_param == 0:
                save_training_plots(history)

            if completed_episodes >= episodes:
                break

    pbar.close()
    eval_env.close()
    save_training_plots(history)
    agent.save('models/final_model.pth')
    envs.close()
    wandb.finish()


def test_policy(model_path, agent=None, env=None, frame_skip=4, eval_epsilon=0.0):
    """Load a trained model and run it in the environment with human rendering.

    eval_epsilon controls exploration during evaluation:
      - 0.0 => fully greedy policy
      - >0  => epsilon-greedy evaluation
    """
    model_path = resolve_model_path(model_path)
    print(f"Loading model from: {model_path}")
    agent.load(model_path)

    obs, info = env.reset()
    total_reward = 0
    episode_over = False

    while not episode_over:
        if np.random.random() < eval_epsilon:
            action = env.action_space.sample()
        else:
            action = agent.greedy_action(obs)
        for _ in range(frame_skip):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_over = terminated or truncated
            if episode_over:
                break

    env.close()
    print(f"Episode finished with reward: {total_reward}")


def resolve_model_path(model_path):
    """Resolve common model-path variants for eval mode.

    Supports:
    - missing extension (.pth/.pt)
    - models/ and best_model/ prefixes
    - legacy shorthand like best_model_27000.pt
    """
    if os.path.exists(model_path):
        return model_path

    stem, ext = os.path.splitext(model_path)
    name_candidates = [model_path]
    if ext == "":
        name_candidates.extend([f"{stem}.pth", f"{stem}.pt"])
    elif ext == ".pt":
        name_candidates.append(f"{stem}.pth")
    elif ext == ".pth":
        name_candidates.append(f"{stem}.pt")

    base_name = os.path.basename(stem if ext else model_path)
    if base_name.startswith("best_model_"):
        suffix = base_name[len("best_model_"):]
        for e in (".pth", ".pt"):
            name_candidates.append(f"dqn_agent_{suffix}{e}")

    search_dirs = ["", "models", "best_model"]
    tried = []
    for candidate_name in name_candidates:
        for directory in search_dirs:
            candidate_path = os.path.join(directory, candidate_name) if directory else candidate_name
            if candidate_path in tried:
                continue
            tried.append(candidate_path)
            if os.path.exists(candidate_path):
                return candidate_path

    available = []
    for directory in ("models", "best_model"):
        if not os.path.isdir(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".pth") or filename.endswith(".pt"):
                available.append(os.path.join(directory, filename))

    msg = [f"Model file not found: {model_path}", "Tried:"]
    msg.extend(f"  - {p}" for p in tried)
    if available:
        msg.append("Available checkpoints:")
        msg.extend(f"  - {p}" for p in available[:10])
    raise FileNotFoundError("\n".join(msg))


def record_demos(env, frame_skip=4, save_path='demos/human_demos.npz'):
    """Play MsPacman with arrow keys and record transitions to disk.

    Uses its own pygame window (via render_mode='rgb_array') so that
    keyboard input is always captured reliably.

    Key mapping (MsPacman-v5 action space):
        Arrow keys        → UP / RIGHT / LEFT / DOWN
        Two arrows at once→ diagonal (UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT)
        No key            → NOOP
        Q                 → quit and save
    """
    import pygame
    pygame.init()

    DISPLAY_SCALE = 3
    base_w, base_h = 160, 210
    screen = pygame.display.set_mode(
        (base_w * DISPLAY_SCALE, base_h * DISPLAY_SCALE))
    pygame.display.set_caption("MsPacman — Recording Demos")
    clock = pygame.time.Clock()

    obs, _ = env.reset()

    states, actions, rewards, next_states, dones = [], [], [], [], []
    total_transitions = 0
    episode = 0
    episode_reward = 0.0

    print("=== Human Demo Recorder ===")
    print("Arrow keys to move, Q to stop and save.\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        if not running:
            break

        keys = pygame.key.get_pressed()
        up = keys[pygame.K_UP]
        down = keys[pygame.K_DOWN]
        left = keys[pygame.K_LEFT]
        right = keys[pygame.K_RIGHT]

        if up and right:
            action = 5
        elif up and left:
            action = 6
        elif down and right:
            action = 7
        elif down and left:
            action = 8
        elif up:
            action = 1
        elif right:
            action = 2
        elif left:
            action = 3
        elif down:
            action = 4
        else:
            action = 0

        reward_sum = 0.0
        done = False
        for _ in range(frame_skip):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            done = terminated or truncated

            frame = env.render()
            if frame is not None:
                surf = pygame.surfarray.make_surface(
                    np.transpose(frame, (1, 0, 2)))
                scaled = pygame.transform.scale(
                    surf, (base_w * DISPLAY_SCALE, base_h * DISPLAY_SCALE))
                screen.blit(scaled, (0, 0))
                pygame.display.flip()
            clock.tick(15)

            if done:
                break

        states.append(np.array(obs, dtype=np.uint8))
        actions.append(action)
        rewards.append(reward_sum)
        next_states.append(np.array(next_obs, dtype=np.uint8))
        dones.append(float(done))

        total_transitions += 1
        episode_reward += reward_sum
        obs = next_obs

        print(f"\rTransitions: {total_transitions}  "
              f"Episode: {episode + 1}  "
              f"Score: {episode_reward:.0f}",
              end="", flush=True)

        if done:
            episode += 1
            print(f"\n  Episode {episode} finished — score {episode_reward:.0f}")
            episode_reward = 0.0
            obs, _ = env.reset()

        pygame.event.pump()  # keep the window responsive between decisions

    pygame.quit()
    env.close()

    if total_transitions == 0:
        print("\nNo transitions recorded, nothing to save.")
        return

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    np.savez_compressed(
        save_path,
        states=np.array(states),
        actions=np.array(actions),
        rewards=np.array(rewards, dtype=np.float32),
        next_states=np.array(next_states),
        dones=np.array(dones, dtype=np.float32),
    )
    print(f"\nSaved {total_transitions} transitions to {save_path}")


def make_agent(obs_shape, n_actions, config):
    """Create a DQNAgent from the config dict."""
    return DQNAgent(
        obs_shape,
        n_actions,
        memory_size=config['agent']['memory_size'],
        batch_size=config['agent']['batch_size'],
        gamma=config['agent']['gamma'],
        epsilon=config['agent']['epsilon'],
        epsilon_min=config['agent']['epsilon_min'],
        epsilon_decay=config['agent']['epsilon_decay'],
        learning_rate=config['agent']['learning_rate'],
        target_update_frequency=config['agent']['target_update_frequency'],
        tau=config['agent'].get('tau', 0.005),
        target_update_mode=config['agent'].get('target_update_mode', 'soft'),
        demo_sample_prob=config['agent'].get('demo_sample_prob', 0.0),
        per_alpha=config['agent'].get('per_alpha', 0.6),
        per_beta_start=config['agent'].get('per_beta_start', 0.4),
        per_beta_end=config['agent'].get('per_beta_end', 1.0),
        per_beta_steps=config['agent'].get('per_beta_steps', 1000000),
        per_eps=config['agent'].get('per_eps', 1e-5),
        grad_clip_norm=config['agent'].get('grad_clip_norm', 1.0),
    )


def main():
    parser = argparse.ArgumentParser(description="Pac-Man DQN Agent")
    parser.add_argument("mode", choices=["train", "eval", "record"],
                        help="'train' | 'eval' | 'record'")
    parser.add_argument("--model", type=str, default="models/final_model.pth",
                        help="model checkpoint for eval "
                             "(default: models/final_model.pth)")
    parser.add_argument("--eval-epsilon", type=float, default=None,
                        help="epsilon used in eval mode "
                             "(CLI overrides hyperparameters.yaml)")
    parser.add_argument("--demos", type=str, default=None,
                        help="path to .npz demo file to pre-load into "
                             "the replay buffer before training")
    parser.add_argument("--save-path", type=str,
                        default="demos/human_demos.npz",
                        help="where to save recorded demos "
                             "(default: demos/human_demos.npz)")
    args = parser.parse_args()

    config = load_hyperparameters()
    gym.register_envs(ale_py)

    if args.mode == "train":
        envs = create_envs(
            num_envs=config['environment']['num_envs'],
            render_mode=config['environment']['render_mode'],
            stack_size=config['environment']['stack_size'],
            resize_shape=config['environment']['resize_shape'],
        )
        agent = make_agent(
            envs.single_observation_space.shape,
            envs.single_action_space.n, config)

        demo_prob = config['agent'].get('demo_sample_prob', 0.0)
        if demo_prob > 0 and not args.demos:
            parser.error(
                f"demo_sample_prob is {demo_prob} but no --demos file was "
                "provided. Either pass --demos <path> or set "
                "demo_sample_prob to 0 in hyperparameters.yaml")

        if args.demos:
            agent.load_demos(args.demos)

        train(
            episodes=config['training']['episodes'],
            agent=agent,
            envs=envs,
            frame_skip=config['environment']['stack_size'],
            avg_param=config['training']['avg_param'],
            eval_interval=config['training'].get('eval_interval', 100),
            eval_episodes=config['training'].get('eval_episodes', 10),
            train_steps_per_sim_step=config['training'].get(
                'train_steps_per_sim_step', 4),
            shaping_alpha=config['training'].get('shaping_alpha', 0.01),
            shaping_gamma=config['training'].get('shaping_gamma', 0.99),
            reward_scale=config['training'].get('reward_scale', 1.0),
            life_loss_penalty_scale=config['training'].get(
                'life_loss_penalty_scale', 1.0),
            config=config,
        )

    elif args.mode == "eval":
        eval_epsilon = args.eval_epsilon
        if eval_epsilon is None:
            eval_epsilon = config['training'].get('eval_epsilon', 0.0)

        if not (0.0 <= eval_epsilon <= 1.0):
            parser.error("--eval-epsilon must be between 0 and 1")

        env = create_env(
            render_mode='human',
            stack_size=config['environment']['stack_size'],
            resize_shape=config['environment']['resize_shape'],
        )
        agent = make_agent(
            env.observation_space.shape, env.action_space.n, config)
        test_policy(
            model_path=args.model,
            agent=agent,
            env=env,
            frame_skip=config['environment']['stack_size'],
            eval_epsilon=eval_epsilon,
        )

    elif args.mode == "record":
        env = create_env(
            render_mode='rgb_array',
            stack_size=config['environment']['stack_size'],
            resize_shape=config['environment']['resize_shape'],
        )
        record_demos(
            env=env,
            frame_skip=config['environment']['stack_size'],
            save_path=args.save_path,
        )


if __name__ == "__main__":
    main()