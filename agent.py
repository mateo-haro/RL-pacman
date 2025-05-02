import torch
from torchsummary import summary
import torch.nn.functional as F
from collections import deque
import random
from model import DQN
import numpy as np

class DQNAgent:
    def __init__(self, state_shape, n_actions, 
                 memory_size=50000,  # Reduced from 100000 to 50000
                 batch_size=32,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.1,
                 epsilon_decay=0.995,
                 learning_rate=0.001,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize DQN Agent.
        
        Memory size is set to 50,000 experiences. Each experience contains:
        - state (4 frames of 84x84 grayscale): ~28KB
        - action: 4 bytes
        - reward: 4 bytes
        - next_state: ~28KB
        - done: 1 byte
        
        Total memory usage: ~2.8GB for replay buffer
        """
        self.device = device
        self.n_actions = n_actions

        print(f"Using device: {self.device}")
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize replay buffer
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)

        # Create Q and target networks
        self.q_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_update_frequency = 2000
        self.steps = 0

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        
        # Initialize loss tracking
        self.losses = []
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            # Set network to evaluation mode
            self.q_net.eval()
            # Normalize state to match training
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
            q_values = self.q_net(state)
            # Set network back to training mode
            self.q_net.train()
            return q_values.argmax().item()
    
    def replay_training(self):
        """Train on batch of experiences."""
        if len(self.memory) < 1000: #self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and normalize states
        states = torch.FloatTensor(np.array(states)).to(self.device) / 255.0
        actions = torch.LongTensor(actions).to(self.device)
        # Scale rewards to [-1, 1] range
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device) / 255.0
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values from target net
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        self.losses.append(loss.item())  # Track loss
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
