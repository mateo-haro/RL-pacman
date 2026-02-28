import torch
import torch.nn.functional as F
import random
from model import DQN
import numpy as np


class ReplayBuffer:
    """Memory-efficient replay buffer using pre-allocated numpy arrays.

    Stores states as uint8 to avoid the massive overhead of a Python deque
    holding individual numpy array objects. For 100k entries with state shape
    (4, 84, 84): ~5.3 GB total (predictable, allocated upfront).
    """
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, alpha=0.0, beta=1.0):
        batch_size = min(batch_size, self.size)
        if batch_size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        if alpha <= 0:
            indices = np.random.choice(self.size, batch_size, replace=False)
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            priorities = self.priorities[:self.size]
            probs = priorities ** alpha
            probs_sum = probs.sum()
            if probs_sum <= 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / probs_sum

            indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
            weights = (self.size * probs[indices]) ** (-beta)
            weights = weights / (weights.max() + 1e-8)
            weights = weights.astype(np.float32)

        return (self.states[indices], self.actions[indices],
                self.rewards[indices], self.next_states[indices],
                self.dones[indices], indices, weights)

    def update_priorities(self, indices, td_errors, eps=1e-5):
        if len(indices) == 0:
            return
        new_priorities = np.abs(td_errors) + eps
        self.priorities[indices] = new_priorities
        self.max_priority = max(self.max_priority, float(new_priorities.max()))

    def load_from_file(self, path):
        """Load recorded demonstrations from a .npz file into the buffer."""
        data = np.load(path)
        n = min(len(data["states"]), self.capacity)
        self.states[:n] = data["states"][:n]
        self.next_states[:n] = data["next_states"][:n]
        self.actions[:n] = data["actions"][:n]
        self.rewards[:n] = data["rewards"][:n]
        self.dones[:n] = data["dones"][:n]
        self.priorities[:n] = self.max_priority
        self.pos = n % self.capacity
        self.size = n
        return n

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, state_shape, n_actions,
                 memory_size=50000,
                 batch_size=32,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.1,
                 epsilon_decay=0.995,
                 learning_rate=0.001,
                 target_update_frequency=5000,
                 tau=0.005,
                 target_update_mode="soft",
                 demo_sample_prob=0.0,
                 per_alpha=0.6,
                 per_beta_start=0.4,
                 per_beta_end=1.0,
                 per_beta_steps=1000000,
                 per_eps=1e-5,
                 grad_clip_norm=1.0,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.n_actions = n_actions
        self.state_shape = state_shape

        print(f"Using device: {self.device}")
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.target_update_mode = target_update_mode.lower()
        if self.target_update_mode not in {"hard", "soft"}:
            raise ValueError(
                "target_update_mode must be 'hard' or 'soft', "
                f"got: {target_update_mode}"
            )
        self.demo_sample_prob = demo_sample_prob
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end
        self.per_beta_steps = per_beta_steps
        self.per_eps = per_eps
        self.grad_clip_norm = grad_clip_norm

        # Memory-efficient replay buffer (pre-allocated numpy arrays)
        self.memory = ReplayBuffer(memory_size, state_shape)
        self.demo_buffer = None

        # Create Q and target networks
        self.q_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_update_frequency = target_update_frequency
        self.steps = 0

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.losses = []

    def load_demos(self, path):
        """Load recorded demonstrations into a dedicated demo buffer."""
        data = np.load(path)
        n = len(data["states"])
        self.demo_buffer = ReplayBuffer(n, self.state_shape)
        loaded = self.demo_buffer.load_from_file(path)
        print(f"Loaded {loaded} demo transitions into separate demo buffer")
        return loaded

    def remember(self, states, actions, rewards, next_states, dones):
        """Store experiences from multiple environments in replay buffer."""
        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i],
                            next_states[i], dones[i])

    def act(self, states):
        """Choose actions for multiple environments using epsilon-greedy policy.

        Batches all exploit-mode states through the network in a single
        forward pass instead of looping one-by-one.
        """
        batch_size = len(states)
        actions = np.zeros(batch_size, dtype=np.int64)

        # Per-env epsilon-greedy decision
        explore_mask = np.random.random(batch_size) < self.epsilon
        actions[explore_mask] = np.random.randint(
            0, self.n_actions, size=int(explore_mask.sum()))

        exploit_indices = np.where(~explore_mask)[0]
        if len(exploit_indices) > 0:
            with torch.no_grad():
                self.q_net.eval()
                exploit_states = torch.FloatTensor(
                    np.array(states)[exploit_indices]
                ).to(self.device) / 255.0
                q_values = self.q_net(exploit_states)
                self.q_net.train()
                actions[exploit_indices] = q_values.argmax(dim=1).cpu().numpy()

        return actions

    def greedy_action(self, state):
        """Choose a single greedy action for evaluation (no exploration)."""
        with torch.no_grad():
            self.q_net.eval()
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
            q_values = self.q_net(state_t)
            self.q_net.train()
            return q_values.argmax().item()

    def _current_per_beta(self):
        progress = min(1.0, self.steps / max(1, self.per_beta_steps))
        return self.per_beta_start + progress * (self.per_beta_end - self.per_beta_start)

    def _grad_norm(self):
        total_sq = 0.0
        for param in self.q_net.parameters():
            if param.grad is None:
                continue
            g = param.grad.detach()
            total_sq += g.pow(2).sum().item()
        return total_sq ** 0.5

    def replay_training(self):
        """Train on a batch and return diagnostic metrics.

        When demo_sample_prob > 0 and a demo buffer is loaded, a fraction
        of each batch is drawn from the demo buffer and the rest from the
        live replay buffer.

        Returns None if the buffer is too small, otherwise a dict with:
        loss, mean_td_error, mean_q, max_q.
        """
        if len(self.memory) < 1000:
            return None

        use_demos = (self.demo_buffer is not None
                     and len(self.demo_buffer) > 0
                     and self.demo_sample_prob > 0)

        beta = self._current_per_beta()
        priority_updates = []

        if use_demos:
            n_demo = int(self.batch_size * self.demo_sample_prob)
            n_demo = min(n_demo, len(self.demo_buffer))
            n_live = self.batch_size - n_demo

            sampled = []
            cursor = 0
            if n_demo > 0:
                d_s, d_a, d_r, d_ns, d_d, d_idx, d_w = self.demo_buffer.sample(
                    n_demo, alpha=self.per_alpha, beta=beta)
                sampled.append((d_s, d_a, d_r, d_ns, d_d, d_w))
                priority_updates.append(
                    (self.demo_buffer, d_idx, cursor, cursor + len(d_idx)))
                cursor += len(d_idx)

            if n_live > 0:
                l_s, l_a, l_r, l_ns, l_d, l_idx, l_w = self.memory.sample(
                    n_live, alpha=self.per_alpha, beta=beta)
                sampled.append((l_s, l_a, l_r, l_ns, l_d, l_w))
                priority_updates.append(
                    (self.memory, l_idx, cursor, cursor + len(l_idx)))
                cursor += len(l_idx)

            states = np.concatenate([x[0] for x in sampled], axis=0)
            actions = np.concatenate([x[1] for x in sampled], axis=0)
            rewards = np.concatenate([x[2] for x in sampled], axis=0)
            next_states = np.concatenate([x[3] for x in sampled], axis=0)
            dones = np.concatenate([x[4] for x in sampled], axis=0)
            is_weights = np.concatenate([x[5] for x in sampled], axis=0).astype(np.float32)
        else:
            (states, actions, rewards, next_states, dones,
             indices, is_weights) = self.memory.sample(
                self.batch_size, alpha=self.per_alpha, beta=beta)
            priority_updates.append((self.memory, indices, 0, len(indices)))

        states = torch.FloatTensor(states).to(self.device) / 255.0
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device) / 255.0
        dones = torch.FloatTensor(dones).to(self.device)
        is_weights = torch.FloatTensor(is_weights).to(self.device)

        all_q_values = self.q_net(states)
        current_q_values = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # next_q_values = self.target_net(next_states).max(1)[0]
            # Double DQN
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)).squeeze(1)

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = (target_q_values - current_q_values).abs()
        sample_losses = F.smooth_l1_loss(
            current_q_values, target_q_values, reduction='none')
        loss = (is_weights * sample_losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = self._grad_norm()
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        td_errors_np = td_errors.detach().cpu().numpy()
        if self.per_alpha > 0:
            for buffer, indices, start, end in priority_updates:
                buffer.update_priorities(
                    indices, td_errors_np[start:end], eps=self.per_eps)

        self.steps += 1
        if self.target_update_mode == "hard":
            if self.steps % self.target_update_frequency == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            # Polyak (soft) update: target ← τ·q_net + (1−τ)·target
            target_sd = self.target_net.state_dict()
            for key, param in self.q_net.state_dict().items():
                target_sd[key].copy_(
                    self.tau * param + (1 - self.tau) * target_sd[key])
            self.target_net.load_state_dict(target_sd)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        metrics = {
            "loss": loss.item(),
            "mean_td_error": td_errors.mean().item(),
            "mean_q": current_q_values.mean().item(),
            "max_q": all_q_values.max().item(),
            "grad_norm": grad_norm,
            "mean_is_weight": is_weights.mean().item(),
            "min_is_weight": is_weights.min().item(),
            "max_is_weight": is_weights.max().item(),
        }
        self.losses.append(metrics["loss"])
        return metrics

    def save(self, path):
        """Save model to file."""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        
    def load(self, path):
        """Load model from file."""
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
