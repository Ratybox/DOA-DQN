"""
DOA-DQN Hyperparameter Optimization for Atari (ALE)
====================================================
Dream Optimization Algorithm applied to Deep Q-Network hyperparameter tuning
Environment: Atari-57 (Arcade Learning Environment)

SOTA Baselines for Atari-57:
- DQN: 121.9 mean HNS (Mnih et al., 2016)
- Double DQN: 332.9 mean HNS
- Rainbow: 223.0 median HNS (Hessel et al., 2017/2018)

References:
[1] Mnih et al. (2016): Asynchronous Methods for Deep Reinforcement Learning
[2] Hessel et al. (2017/2018): Rainbow: Combining Improvements in Deep RL
[3] AutoDO Paper (2024): Automated Decision Optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Callable, Tuple, Optional, Dict, List
import warnings
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import cv2
import subprocess
import sys

warnings.filterwarnings('ignore')

# Try to import gym and install ale-py if needed
try:
    import gym
except ImportError:
    print("Installing gym...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gym"])
    import gym

# Install and import ale-py for Atari environments
try:
    import ale_py
    from ale_py import ALEInterface
except ImportError:
    print("Installing ale-py for Atari environments...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ale-py"])
    import ale_py
    from ale_py import ALEInterface

# Register Atari environments
try:
    from ale_py.roms import Breakout
    ale_py.import_roms()
except Exception as e:
    print(f"Warning: Could not import ROMs automatically: {e}")
    print("You may need to run: pip install gym[atari] or ale-py-roms")

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("✓ Imports successful")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ NumPy version: {np.__version__}")

# Check if Atari environments are available
try:
    test_env = gym.make("ALE/Breakout-v5")
    test_env.close()
    print("✓ Atari environment (ALE/Breakout-v5) is available")
except Exception as e:
    print(f"⚠ Atari environment not available: {e}")
    print("  Trying to install ale-py...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ale-py", "--quiet"])
        print("✓ ale-py installed successfully")
    except Exception as install_error:
        print(f"⚠ Could not install ale-py automatically: {install_error}")
        print("  Please run manually: pip install ale-py gym[atari]")
        print("  Or use alternative environment: Breakout-v4")


# ============================================================================
# REPLAY BUFFER
# ============================================================================
class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch from buffer."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

print("✓ ReplayBuffer class defined")


# ============================================================================
# NEURAL NETWORK (CNN for Atari)
# ============================================================================
class QNetwork(nn.Module):
    """CNN Q-network for Atari games (following DQN architecture)."""
    
    def __init__(self, action_dim: int, n_frames: int = 4):
        super(QNetwork, self).__init__()
        # Input: (batch, n_frames, 84, 84) after preprocessing
        self.conv1 = nn.Conv2d(n_frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate flattened size: 64 * 7 * 7 = 3136 (for 84x84 input)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)
    
    def forward(self, x):
        # x shape: (batch, n_frames, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

print("✓ QNetwork (CNN) class defined")


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Preprocess Atari frame: grayscale, resize to 84x84, normalize."""
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Resize to 84x84
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    return frame

# ============================================================================
# DQN AGENT (Atari)
# ============================================================================
class DQNAgent:
    """Deep Q-Network Agent for Atari games with CNN."""
    
    def __init__(
        self,
        env: gym.Env,
        hyperparams: Dict,
        device: str = 'cpu',
        seed: int = 42,
        n_frames: int = 4
    ):
        """
        Initialize DQN Agent for Atari.
        
        Parameters
        ----------
        env : gym.Env
            Gym environment (Atari)
        hyperparams : Dict
            Dictionary containing all hyperparameters
        device : str
            'cpu' or 'cuda'
        seed : int
            Random seed
        n_frames : int
            Number of frames to stack (default: 4)
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.env = env
        self.device = device
        self.hyperparams = hyperparams
        self.n_frames = n_frames
        
        # Action space
        self.action_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else 1
        
        # Networks (CNN for Atari)
        self.q_network = QNetwork(
            action_dim=self.action_dim,
            n_frames=n_frames
        ).to(device)
        
        self.target_network = QNetwork(
            action_dim=self.action_dim,
            n_frames=n_frames
        ).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer with Adam hyperparameters
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=hyperparams['learning_rate'],
            betas=(hyperparams['adam_beta1'], hyperparams['adam_beta2']),
            eps=hyperparams['adam_epsilon']
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(int(hyperparams['replay_buffer_size']))
        
        # Hyperparameters
        self.gamma = hyperparams['gamma']
        self.epsilon_start = hyperparams['epsilon_start']
        self.epsilon_end = hyperparams['epsilon_end']
        self.epsilon_decay_steps = hyperparams['epsilon_decay_steps']
        self.batch_size = int(hyperparams['batch_size'])
        self.target_update_freq = int(hyperparams['target_update_freq'])
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.training_losses = []
        
        # Frame buffer for stacking
        self.frame_buffer = deque(maxlen=n_frames)
    
    def _preprocess_and_stack_frames(self, frame: np.ndarray, reset: bool = False) -> np.ndarray:
        """Preprocess frame and stack with previous frames."""
        processed = preprocess_frame(frame)
        
        if reset:
            self.frame_buffer.clear()
            # Fill buffer with same frame on reset
            for _ in range(self.n_frames):
                self.frame_buffer.append(processed)
        else:
            self.frame_buffer.append(processed)
        
        # Stack frames: shape (n_frames, 84, 84)
        stacked = np.stack(list(self.frame_buffer), axis=0)
        return stacked
    
    def get_epsilon(self) -> float:
        """Get current epsilon for epsilon-greedy exploration."""
        progress = min(self.step_count / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress
    
    def select_action(self, stacked_frames: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.rand() < self.get_epsilon():
            return self.env.action_space.sample()
        
        with torch.no_grad():
            # stacked_frames shape: (n_frames, 84, 84)
            # Convert to (1, n_frames, 84, 84) for batch
            state_tensor = torch.FloatTensor(stacked_frames).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(1).item()
    
    def update(self):
        """Update Q-network with batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # States shape: (batch, n_frames, 84, 84)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q(s,a)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q(s',a')
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # TD loss
        loss = nn.MSELoss()(q_values, target_q)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_losses.append(loss.item())
        
        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self, num_steps: int, log_interval: int = 10000):
        """
        Train agent for specified number of steps.
        
        Parameters
        ----------
        num_steps : int
            Total steps to train
        log_interval : int
            Logging interval
        """
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            raw_frame = reset_result[0]
        else:
            raw_frame = reset_result
        
        # Preprocess and stack initial frames
        stacked_state = self._preprocess_and_stack_frames(raw_frame, reset=True)
        episode_reward = 0
        
        for step in range(num_steps):
            # Select and execute action
            action = self.select_action(stacked_state, training=True)
            step_result = self.env.step(action)
            
            if len(step_result) == 5:
                next_raw_frame, reward, terminated, truncated, _ = step_result
            elif len(step_result) == 4:
                next_raw_frame, reward, done, info = step_result
                terminated = done
                truncated = False
            else:
                next_raw_frame, reward, done = step_result[:3]
                terminated = done
                truncated = False
            
            done = terminated or truncated
            
            # Preprocess next frame and stack
            next_stacked_state = self._preprocess_and_stack_frames(next_raw_frame, reset=False)
            
            # Store experience (stacked frames)
            self.replay_buffer.push(stacked_state, action, reward, next_stacked_state, done)
            
            # Update network
            self.update()
            
            episode_reward += reward
            self.step_count += 1
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_count += 1
                
                # Reset environment
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    raw_frame = reset_result[0]
                else:
                    raw_frame = reset_result
                stacked_state = self._preprocess_and_stack_frames(raw_frame, reset=True)
                episode_reward = 0
            else:
                stacked_state = next_stacked_state
            
            if (step + 1) % log_interval == 0 and len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Step {step + 1}/{num_steps} | Avg Reward (last 10): {avg_reward:.2f}")
    
    def evaluate(self, num_episodes: int = 10) -> float:
        """
        Evaluate agent (greedy policy).
        
        Parameters
        ----------
        num_episodes : int
            Number of evaluation episodes
        
        Returns
        -------
        mean_reward : float
            Average reward over episodes
        """
        rewards = []
        
        for _ in range(num_episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                raw_frame = reset_result[0]
            else:
                raw_frame = reset_result
            
            stacked_state = self._preprocess_and_stack_frames(raw_frame, reset=True)
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(stacked_state, training=False)
                step_result = self.env.step(action)
                
                if len(step_result) == 5:
                    raw_frame, reward, terminated, truncated, _ = step_result
                elif len(step_result) == 4:
                    raw_frame, reward, done, info = step_result
                    terminated = done
                    truncated = False
                else:
                    raw_frame, reward, done = step_result[:3]
                    terminated = done
                    truncated = False
                
                done = terminated or truncated
                stacked_state = self._preprocess_and_stack_frames(raw_frame, reset=False)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        return np.mean(rewards)

print("✓ DQNAgent class defined")


# ============================================================================
# DREAM OPTIMIZATION ALGORITHM (DOA)
# ============================================================================
class DOA:
    """Dream Optimization Algorithm for hyperparameter search."""
    
    def __init__(
        self,
        dim: int,
        bounds: list,
        pop_size: int = 20,
        max_iter: int = 100,
        u: float = 0.9,
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        self.dim = dim
        self.bounds = np.array(bounds, dtype=float)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.u = u
        self.verbose = verbose
        
        assert self.bounds.shape == (dim, 2), "Bounds must have shape (dim, 2)"
        assert np.all(self.bounds[:, 0] < self.bounds[:, 1]), "Lower bounds must be < upper bounds"
        
        self.rng = np.random.RandomState(seed)
        self.T_d = int(0.9 * max_iter)
        
        self.k_q = self._compute_k_q()
        self.k_r = self._compute_k_r()
        
        self.history_best = []
        self.history_mean = []
        self.n_evals = 0
    
    def _compute_k_q(self) -> np.ndarray:
        """Compute k_q parameters for exploration phase."""
        k_q = np.zeros(5, dtype=int)
        for q in range(5):
            q_id = q + 1
            k_min = int(np.ceil(self.dim / (8.0 * q_id)))
            k_max = max(2, int(np.ceil(self.dim / (3.0 * q_id))))
            k_q[q] = self.rng.randint(k_min, k_max + 1)
        return k_q
    
    def _compute_k_r(self) -> int:
        """Compute k_r parameter for exploitation phase."""
        k_min = 2
        k_max = max(2, int(np.ceil(self.dim / 3.0)))
        return self.rng.randint(k_min, k_max + 1)
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize population uniformly in [0, 1]^dim."""
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        return low + self.rng.rand(self.pop_size, self.dim) * (high - low)
    
    def _clip_bounds(self, X: np.ndarray) -> np.ndarray:
        """Clip solutions to bounds."""
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        return np.clip(X, low, high)
    
    def _exploration_phase(self, X: np.ndarray, fitness: np.ndarray, t: int) -> np.ndarray:
        """Exploration phase with memory retention and forgetting."""
        X_new = X.copy()
        pop_per_group = self.pop_size // 5
        
        for q in range(5):
            start_idx = q * pop_per_group
            end_idx = self.pop_size if q == 4 else (q + 1) * pop_per_group
            group_indices = np.arange(start_idx, end_idx)
            
            group_fitness = fitness[group_indices]
            best_group_idx = group_indices[np.argmin(group_fitness)]
            X_best_group = X[best_group_idx].copy()
            
            for i in group_indices:
                X_new[i] = X_best_group.copy()
                
                k = self.k_q[q]
                forgetting_dims = self.rng.choice(self.dim, size=min(k, self.dim), replace=False)
                
                for j in forgetting_dims:
                    if self.rng.rand() < self.u:
                        low, high = self.bounds[j]
                        perturb = low + self.rng.rand() * (high - low)
                        alpha = 0.5 * (np.cos(np.pi * (t + self.max_iter - self.T_d) / self.max_iter) + 1.0)
                        X_new[i, j] = X_best_group[j] + (perturb - X_best_group[j]) * alpha
                    else:
                        m = self.rng.randint(0, self.pop_size)
                        X_new[i, j] = X_new[m, j] if m <= i else X[m, j]
        
        return X_new
    
    def _exploitation_phase(self, X: np.ndarray, X_best: np.ndarray, t: int) -> np.ndarray:
        """Exploitation phase focused on local search."""
        X_new = np.tile(X_best, (self.pop_size, 1))
        
        k = self.k_r
        forgetting_dims = self.rng.choice(self.dim, size=min(k, self.dim), replace=False)
        
        for i in range(self.pop_size):
            for j in forgetting_dims:
                low, high = self.bounds[j]
                perturb = low + self.rng.rand() * (high - low)
                alpha = 0.5 * (np.cos(np.pi * t / self.max_iter) + 1.0)
                X_new[i, j] = X_best[j] + (perturb - X_best[j]) * alpha
        
        return X_new
    
    def optimize(self, fitness_fn: Callable) -> Tuple[np.ndarray, float, Dict]:
        """Run DOA optimization."""
        X = self._initialize_population()
        fitness = np.array([fitness_fn(x) for x in X])
        self.n_evals = self.pop_size
        
        # Check if all evaluations failed
        valid_fitness = fitness[np.isfinite(fitness)]
        if len(valid_fitness) == 0:
            raise RuntimeError(
                "All fitness evaluations failed! "
                "Please check your environment setup and ensure ale-py is installed: "
                "pip install ale-py gym[atari]"
            )
        
        best_idx = np.argmin(fitness)
        X_best = X[best_idx].copy()
        f_best = fitness[best_idx]
        
        self.history_best.append(f_best)
        self.history_mean.append(np.mean(fitness))
        
        if self.verbose:
            print(f"[DOA] Iter 0 - Best: {f_best:.6e}, Mean: {np.mean(fitness):.6e}")
        
        for t in range(self.max_iter):
            if t < self.T_d:
                X_new = self._exploration_phase(X, fitness, t)
            else:
                X_new = self._exploitation_phase(X, X_best, t)
            
            X_new = self._clip_bounds(X_new)
            fitness_new = np.array([fitness_fn(x) for x in X_new])
            self.n_evals += self.pop_size
            
            improved = fitness_new < fitness
            X[improved] = X_new[improved]
            fitness[improved] = fitness_new[improved]
            
            # Only update best if we have valid fitness values
            valid_fitness = fitness[np.isfinite(fitness)]
            if len(valid_fitness) > 0:
                best_idx = np.argmin(fitness)
                
                if fitness[best_idx] < f_best:
                    f_best = fitness[best_idx]
                    X_best = X[best_idx].copy()
            # If all failed, keep previous best and continue
            
            self.history_best.append(f_best)
            self.history_mean.append(np.mean(fitness))
            
            if self.verbose and (t + 1) % max(1, self.max_iter // 10) == 0:
                print(f"[DOA] Iter {t+1} - Best: {f_best:.6e}, Mean: {np.mean(fitness):.6e}")
        
        info = {
            'history_best': np.array(self.history_best),
            'history_mean': np.array(self.history_mean),
            'n_evals': self.n_evals,
        }
        
        return X_best, f_best, info

print("✓ DOA class defined")


# ============================================================================
# HYPERPARAMETER CODEC
# ============================================================================
class HyperparameterCodec:
    """Encode/decode hyperparameters from/to continuous vectors."""
    
    # Hyperparameter specifications for Atari (name, lower, upper, type, scale)
    # Note: CNN architecture is fixed, so no hidden_size or n_layers
    SPECS = {
        'learning_rate': (1e-5, 1e-3, 'float', 'log'),  # Lower for Atari
        'gamma': (0.95, 0.999, 'float', 'linear'),
        'epsilon_start': (0.9, 1.0, 'float', 'linear'),
        'epsilon_end': (0.01, 0.1, 'float', 'linear'),
        'epsilon_decay_steps': (100000, 1000000, 'int', 'log'),  # Longer for Atari
        'batch_size': (32, 128, 'int', 'discrete'),  # 32, 64, 128
        'replay_buffer_size': (100000, 1000000, 'int', 'log'),  # Larger for Atari
        'target_update_freq': (1000, 10000, 'int', 'log'),  # Less frequent for Atari
        'adam_beta1': (0.9, 0.99, 'float', 'linear'),
        'adam_beta2': (0.99, 0.9999, 'float', 'linear'),
        'adam_epsilon': (1e-8, 1e-5, 'float', 'log'),
    }
    
    HP_NAMES = list(SPECS.keys())
    DIM = len(HP_NAMES)
    
    @classmethod
    def get_bounds(cls) -> List[Tuple[float, float]]:
        """Get bounds for DOA [0, 1] normalized."""
        return [(0.0, 1.0) for _ in range(cls.DIM)]
    
    @classmethod
    def decode(cls, x: np.ndarray) -> Dict:
        """
        Decode normalized vector [0,1]^D to hyperparameters.
        
        Parameters
        ----------
        x : np.ndarray
            Normalized vector of shape (D,)
        
        Returns
        -------
        hyperparams : Dict
            Dictionary of hyperparameters
        """
        assert len(x) == cls.DIM, f"Expected {cls.DIM} dimensions, got {len(x)}"
        assert np.all((x >= 0) & (x <= 1)), "Input must be in [0, 1]"
        
        hyperparams = {}
        
        for i, hp_name in enumerate(cls.HP_NAMES):
            lower, upper, hp_type, scale = cls.SPECS[hp_name]
            x_norm = x[i]  # value in [0, 1]
            
            if scale == 'log':
                # Log scale
                hp_value = lower * (upper / lower) ** x_norm
            elif scale == 'linear':
                # Linear scale
                hp_value = lower + x_norm * (upper - lower)
            elif scale == 'discrete':
                # Discrete values (powers of 2 or fixed list)
                if hp_name == 'batch_size':
                    # [32, 64, 128] for Atari
                    idx = int(x_norm * 2.99)
                    hp_value = [32, 64, 128][idx]
                else:
                    idx = int(x_norm * (upper - lower + 0.99))
                    hp_value = lower + idx
            else:
                hp_value = lower + x_norm * (upper - lower)
            
            hyperparams[hp_name] = float(hp_value) if hp_type == 'float' else int(hp_value)
        
        return hyperparams
    
    @classmethod
    def validate(cls, hyperparams: Dict) -> bool:
        """Validate hyperparameters."""
        for name, value in hyperparams.items():
            if name not in cls.SPECS:
                return False
            lower, upper, _, _ = cls.SPECS[name]
            if value < lower or value > upper:
                return False
        return True

print("✓ HyperparameterCodec class defined")
print(f" - Total hyperparameters: {HyperparameterCodec.DIM}")
print(f" - Hyperparameters: {', '.join(HyperparameterCodec.HP_NAMES)}")


# ============================================================================
# DOA-DQN OPTIMIZER
# ============================================================================
class DOADQNOptimizer:
    """Main optimizer combining DOA with DQN hyperparameter tuning."""
    
    def __init__(
        self,
        env_name: str = 'ALE/Breakout-v5',
        doa_pop_size: int = 10,
        doa_iterations: int = 5,
        dqn_train_steps: int = 50000,
        dqn_eval_episodes: int = 10,
        device: str = 'cpu',
        output_dir: str = './results',
        verbose: bool = True
    ):
        """
        Initialize DOA-DQN optimizer.
        
        Parameters
        ----------
        env_name : str
            Gym environment name
        doa_pop_size : int
            DOA population size
        doa_iterations : int
            DOA iterations
        dqn_train_steps : int
            DQN training steps per evaluation
        dqn_eval_episodes : int
            Episodes for final DQN evaluation
        device : str
            'cpu' or 'cuda'
        output_dir : str
            Directory for saving results
        verbose : bool
            Verbosity flag
        """
        self.env_name = env_name
        self.doa_pop_size = doa_pop_size
        self.doa_iterations = doa_iterations
        self.dqn_train_steps = dqn_train_steps
        self.dqn_eval_episodes = dqn_eval_episodes
        self.device = device
        self.verbose = verbose
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimization_history = []
    
    def fitness_function(self, x: np.ndarray) -> float:
        """
        Evaluate DQN with given hyperparameters.
        
        Parameters
        ----------
        x : np.ndarray
            Normalized hyperparameter vector
        
        Returns
        -------
        fitness : float
            Negative mean reward (for minimization)
        """
        # Decode hyperparameters
        hyperparams = HyperparameterCodec.decode(x)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Evaluating hyperparameters...")
            for k, v in hyperparams.items():
                print(f"  {k:25s}: {v}")
        
        try:
            # Create environment with fallback options
            try:
                env = gym.make(self.env_name)
            except Exception as e:
                if "ALE" in str(e) or "Namespace" in str(e):
                    # Try alternative environment names
                    alt_envs = [
                        "Breakout-v4",  # Older gym version
                        "Breakout-v0",  # Even older
                        "BreakoutDeterministic-v4",
                        "BreakoutNoFrameskip-v4"
                    ]
                    env = None
                    for alt_env in alt_envs:
                        try:
                            env = gym.make(alt_env)
                            if self.verbose:
                                print(f"Using alternative environment: {alt_env}")
                            break
                        except:
                            continue
                    
                    if env is None:
                        raise RuntimeError(
                            f"Could not create Atari environment. "
                            f"Please install: pip install ale-py gym[atari]"
                        )
                else:
                    raise
            
            # Create DQN agent
            agent = DQNAgent(env, hyperparams, device=self.device, seed=42)
            
            # Train
            if self.verbose:
                print(f"Training for {self.dqn_train_steps} steps...")
            agent.train(self.dqn_train_steps, log_interval=self.dqn_train_steps // 5)
            
            # Evaluate
            if self.verbose:
                print(f"Evaluating on {self.dqn_eval_episodes} episodes...")
            mean_reward = agent.evaluate(self.dqn_eval_episodes)
            
            # Fitness (negative for minimization)
            fitness = -mean_reward
            
            if self.verbose:
                print(f"Mean Reward: {mean_reward:.2f}, Fitness: {fitness:.6e}")
            
            # Store result
            self.optimization_history.append({
                'hyperparams': hyperparams,
                'mean_reward': float(mean_reward),
                'fitness': float(fitness),
                'episode_rewards': agent.episode_rewards[-10:] if len(agent.episode_rewards) >= 10 else agent.episode_rewards,
                'training_losses': agent.training_losses[-100:] if len(agent.training_losses) >= 100 else agent.training_losses,
            })
            
            env.close()
            return fitness
        
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')  # Return worst fitness on error
    
    def optimize(self) -> Tuple[Dict, float, Dict]:
        """
        Run DOA-DQN optimization.
        
        Returns
        -------
        best_hyperparams : Dict
            Best hyperparameters found
        best_reward : float
            Best mean reward achieved
        doa_info : Dict
            DOA optimization information (history, n_evals, etc.)
        """
        print("\n" + "="*70)
        print("Starting DOA-DQN Hyperparameter Optimization for Atari")
        print("="*70)
        print(f"Environment: {self.env_name}")
        print(f"DOA Population: {self.doa_pop_size}")
        print(f"DOA Iterations: {self.doa_iterations}")
        print(f"DQN Training Steps: {self.dqn_train_steps}")
        print("="*70 + "\n")
        
        # Initialize DOA
        doa = DOA(
            dim=HyperparameterCodec.DIM,
            bounds=HyperparameterCodec.get_bounds(),
            pop_size=self.doa_pop_size,
            max_iter=self.doa_iterations,
            seed=42,
            verbose=self.verbose
        )
        
        # Run optimization
        best_x, best_fitness, doa_info = doa.optimize(self.fitness_function)
        
        # Decode best solution
        best_hyperparams = HyperparameterCodec.decode(best_x)
        best_reward = -best_fitness
        
        # Save results
        self._save_results(best_hyperparams, best_reward, doa_info)
        
        print("\n" + "="*70)
        print("Optimization Complete!")
        print(f"Best Reward: {best_reward:.2f}")
        print(f"Total Evaluations: {doa_info['n_evals']}")
        print("="*70 + "\n")
        
        return best_hyperparams, best_reward, doa_info
    
    def _save_results(self, best_hyperparams: Dict, best_reward: float, doa_info: Dict):
        """Save optimization results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"doa_dqn_results_{timestamp}.json"
        
        results = {
            'timestamp': timestamp,
            'env': self.env_name,
            'best_hyperparams': best_hyperparams,
            'best_reward': float(best_reward),
            'optimization_history': self.optimization_history,
            'doa_info': {
                'n_evals': doa_info['n_evals'],
                'history_best': doa_info['history_best'].tolist(),
                'history_mean': doa_info['history_mean'].tolist(),
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"[SAVED] Results to {results_file}")

print("✓ DOADQNOptimizer class defined")


# ============================================================================
# SOTA BASELINES FOR ATARI-57
# ============================================================================
class SOTABaselinesAtari57:
    """Atari-57 (ALE) baselines from literature (Human-Normalized Score)."""
    
    BASELINES = {
        # From Mnih et al. 2016 (A3C) Table 1: 57 games, human-starts metric
        "DQN": {
            "reference": "Mnih et al. (2016)",
            "paper": "Asynchronous Methods for Deep Reinforcement Learning",
            "metric": "HNS_human_starts",
            "mean_hns": 121.9,
            "median_hns": 47.5,
            "notes": "DQN trained ~8 days on GPU (K40) in their comparison table.",
        },
        "DoubleDQN": {
            "reference": "Mnih et al. (2016)",
            "paper": "Asynchronous Methods for Deep Reinforcement Learning",
            "metric": "HNS_human_starts",
            "mean_hns": 332.9,
            "median_hns": 110.9,
        },
        "Dueling_DoubleDQN": {
            "reference": "Mnih et al. (2016)",
            "paper": "Asynchronous Methods for Deep Reinforcement Learning",
            "metric": "HNS_human_starts",
            "mean_hns": 343.8,
            "median_hns": 117.1,
        },
        "Prioritized_DQN": {
            "reference": "Mnih et al. (2016)",
            "paper": "Asynchronous Methods for Deep Reinforcement Learning",
            "metric": "HNS_human_starts",
            "mean_hns": 463.6,
            "median_hns": 127.6,
        },
        "A3C_LSTM": {
            "reference": "Mnih et al. (2016)",
            "paper": "Asynchronous Methods for Deep Reinforcement Learning",
            "metric": "HNS_human_starts",
            "mean_hns": 623.0,
            "median_hns": 112.6,
        },
        # From Hessel et al. (Rainbow) Table 2: median HNS (no-ops / human-starts)
        "Rainbow": {
            "reference": "Hessel et al. (2017/2018)",
            "paper": "Rainbow: Combining Improvements in Deep Reinforcement Learning",
            "metric": "median_HNS",
            "median_hns_noops": 223.0,
            "median_hns_human_starts": 153.0,
            "notes": "Final evaluation snapshot; 57 games; episodes truncated at 108K frames in their setup.",
        },
    }
    
    @classmethod
    def get_sota_baseline(cls, method: str = 'DQN') -> Dict:
        """Get SOTA baseline for comparison."""
        return cls.BASELINES.get(method, cls.BASELINES['DQN'])
    
    @classmethod
    def print_all_baselines(cls):
        """Print all available baselines."""
        print("\n" + "="*70)
        print("SOTA BASELINES FOR ATARI-57 (ALE)")
        print("="*70)
        for method, info in cls.BASELINES.items():
            print(f"\n{method}:")
            print(f"  Reference: {info['reference']}")
            print(f"  Paper: {info['paper']}")
            if 'mean_hns' in info:
                print(f"  Mean HNS: {info['mean_hns']:.1f}")
            if 'median_hns' in info:
                print(f"  Median HNS: {info['median_hns']:.1f}")
            if 'median_hns_noops' in info:
                print(f"  Median HNS (no-ops): {info['median_hns_noops']:.1f}")
            if 'median_hns_human_starts' in info:
                print(f"  Median HNS (human-starts): {info['median_hns_human_starts']:.1f}")
            if 'notes' in info:
                print(f"  Notes: {info['notes']}")
        print("="*70)

print("✓ SOTABaselinesAtari57 class defined")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Auto-detect CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if torch.cuda.is_available():
        print(f"✓ CUDA detected: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Available GPUs: {torch.cuda.device_count()}")
    else:
        print("⚠ CUDA not available, using CPU")
    
    # Print SOTA baselines
    SOTABaselinesAtari57.print_all_baselines()
    
    # Configuration for Atari
    # Try ALE/Breakout-v5 first, fallback to Breakout-v4 if not available
    env_name = "ALE/Breakout-v5"
    try:
        test_env = gym.make(env_name)
        test_env.close()
    except Exception:
        print("⚠ ALE/Breakout-v5 not available, using Breakout-v4")
        env_name = "Breakout-v4"
        try:
            test_env = gym.make(env_name)
            test_env.close()
        except Exception:
            print("⚠ Breakout-v4 not available, trying Breakout-v0")
            env_name = "Breakout-v0"
    
    CONFIG = {
        "env_name": env_name,
        "doa_pop_size": 4,
        "doa_iterations": 3,
        "dqn_train_steps": 2_000_000,   # à augmenter progressivement
        "dqn_eval_episodes": 20,
        "device": device,
        "output_dir": "./doa_dqn_atari_results",
        "verbose": True
    }
    
    print("\nConfiguration:")
    print("="*70)
    for key, value in CONFIG.items():
        print(f" {key:25s}: {value}")
    print("="*70)
    
    # Initialize optimizer
    optimizer = DOADQNOptimizer(**CONFIG)
    
    # Run optimization
    best_hp, best_reward, doa_info = optimizer.optimize()
    
    print(f"\n[SUCCESS] Best DOA-DQN reward: {best_reward:.2f}")
    
    # ========================================================================
    # VISUALIZATION AND COMPARISON
    # ========================================================================
    
    # Plot DOA convergence
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Best fitness over iterations
    axes[0].plot(doa_info['history_best'], 'b-', linewidth=2.5, label='Best Fitness', marker='o', markersize=6)
    axes[0].set_xlabel('DOA Iteration', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Fitness (Negative Reward)', fontsize=12, fontweight='bold')
    axes[0].set_title('DOA Convergence: Best Fitness', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    
    # Mean fitness over iterations
    axes[1].plot(doa_info['history_mean'], 'r-', linewidth=2.5, label='Mean Fitness', marker='s', markersize=6)
    axes[1].set_xlabel('DOA Iteration', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mean Fitness (Negative Reward)', fontsize=12, fontweight='bold')
    axes[1].set_title('DOA Convergence: Mean Fitness', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('doa_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ DOA converged from {doa_info['history_best'][0]:.2f} to {doa_info['history_best'][-1]:.2f}")
    
    # Extract all hyperparameter values and rewards
    hp_data = []
    rewards = []
    
    for entry in optimizer.optimization_history:
        hp_data.append(entry['hyperparams'])
        rewards.append(entry['mean_reward'])
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(hp_data)
    df['reward'] = rewards
    
    # Display best hyperparameters
    print("\n" + "="*70)
    print("BEST HYPERPARAMETERS FOUND BY DOA")
    print("="*70)
    for key, value in sorted(best_hp.items()):
        if isinstance(value, float):
            print(f" {key:25s}: {value:>15.6f}")
        else:
            print(f" {key:25s}: {value:>15}")
    print("="*70)
    
    # Plot hyperparameter distributions
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()
    
    hp_names = HyperparameterCodec.HP_NAMES
    
    for i, hp_name in enumerate(hp_names):
        ax = axes[i]
        values = df[hp_name].values
        
        # Scatter plot: hyperparameter value vs reward
        scatter = ax.scatter(values, df['reward'], alpha=0.7, c=df['reward'],
                           cmap='viridis', s=80, edgecolors='black', linewidth=1)
        ax.axvline(best_hp[hp_name], color='red', linestyle='--', linewidth=2.5, label='Best')
        
        ax.set_xlabel(hp_name, fontsize=10, fontweight='bold')
        ax.set_ylabel('Reward', fontsize=10, fontweight='bold')
        ax.set_title(f'{hp_name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot reward evolution over optimization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Reward over evaluations
    eval_indices = range(1, len(rewards) + 1)
    axes[0].plot(eval_indices, rewards, 'b-', linewidth=2.5, marker='o', markersize=8, label='DOA-DQN')
    
    # Add SOTA baselines for comparison (Atari)
    sota_dqn = SOTABaselinesAtari57.get_sota_baseline('DQN')
    sota_rainbow = SOTABaselinesAtari57.get_sota_baseline('Rainbow')
    
    # Use mean HNS or median HNS for comparison
    sota_dqn_score = sota_dqn.get('mean_hns', sota_dqn.get('median_hns', 0))
    sota_rainbow_score = sota_rainbow.get('median_hns_human_starts', sota_rainbow.get('median_hns_noops', 0))
    
    axes[0].axhline(sota_dqn_score, color='orange', linestyle='--', linewidth=2.5, 
                   label=f"SOTA DQN: {sota_dqn_score:.1f}")
    axes[0].axhline(sota_rainbow_score, color='green', linestyle='--', linewidth=2.5,
                   label=f"SOTA Rainbow: {sota_rainbow_score:.1f}")
    axes[0].axhline(best_reward, color='red', linestyle='--', linewidth=2.5, label=f'Best DOA-DQN: {best_reward:.2f}')
    
    axes[0].set_xlabel('Evaluation Number', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    axes[0].set_title('Reward Evolution During Optimization vs SOTA (Atari)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    # Remove fixed ylim for Atari (rewards vary widely)
    
    # Reward distribution
    axes[1].hist(rewards, bins=min(20, len(rewards)), edgecolor='black', alpha=0.7, color='skyblue', label='DOA-DQN')
    axes[1].axvline(best_reward, color='red', linestyle='--', linewidth=2.5, label=f'Best: {best_reward:.2f}')
    axes[1].axvline(sota_dqn_score, color='orange', linestyle='--', linewidth=2.5, 
                   label=f"SOTA DQN: {sota_dqn_score:.1f}")
    axes[1].axvline(np.mean(rewards), color='green', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(rewards):.2f}')
    
    axes[1].set_xlabel('Mean Reward', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Reward Distribution vs SOTA (Atari)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('reward_comparison_sota.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print reward statistics
    print(f"\n✓ Reward Statistics (DOA-DQN):")
    print(f"  Best: {best_reward:.2f}")
    print(f"  Mean: {np.mean(rewards):.2f}")
    print(f"  Std: {np.std(rewards):.2f}")
    print(f"  Min: {np.min(rewards):.2f}")
    print(f"  Max: {np.max(rewards):.2f}")
    
    # Comparison with SOTA
    print(f"\n✓ Comparison with SOTA (Atari):")
    print(f"  DOA-DQN vs SOTA DQN: {best_reward - sota_dqn_score:+.2f}")
    print(f"  DOA-DQN vs SOTA Rainbow: {best_reward - sota_rainbow_score:+.2f}")
    
    # Find the best configuration's training data
    best_idx = np.argmax(rewards)
    best_training_data = optimizer.optimization_history[best_idx]
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Episode rewards (if available)
    if 'episode_rewards' in best_training_data and len(best_training_data['episode_rewards']) > 0:
        ep_rewards = best_training_data['episode_rewards']
        axes[0].plot(range(len(ep_rewards)), ep_rewards, 'b-', linewidth=2.5, marker='o', markersize=6)
        axes[0].set_xlabel('Episode (last 10)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
        axes[0].set_title('Episode Rewards (Best DOA-DQN Configuration)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        if len(ep_rewards) > 1:
            window = max(1, len(ep_rewards) // 3)
            ma = pd.Series(ep_rewards).rolling(window=window, center=True).mean()
            axes[0].plot(range(len(ma)), ma, 'r--', linewidth=2.5, label=f'MA({window})')
            axes[0].legend(fontsize=11)
    else:
        axes[0].text(0.5, 0.5, 'No episode rewards available',
                    ha='center', va='center', transform=axes[0].transAxes, fontsize=12)
    
    # Training losses (if available)
    if 'training_losses' in best_training_data and len(best_training_data['training_losses']) > 0:
        losses = best_training_data['training_losses']
        axes[1].plot(range(len(losses)), losses, 'g-', linewidth=2.5, alpha=0.7)
        axes[1].set_xlabel('Training Step (last 100)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[1].set_title('Training Loss (Best DOA-DQN Configuration)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        if len(losses) > 10:
            window = max(1, len(losses) // 5)
            ma = pd.Series(losses).rolling(window=window, center=True).mean()
            axes[1].plot(range(len(ma)), ma, 'r--', linewidth=2.5, label=f'MA({window})')
            axes[1].legend(fontsize=11)
    else:
        axes[1].text(0.5, 0.5, 'No training losses available',
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create correlation matrix between hyperparameters and reward
    correlation_data = df.copy()
    numeric_cols = [col for col in correlation_data.columns if correlation_data[col].dtype in ['float64', 'int64']]
    corr_matrix = correlation_data[numeric_cols].corr()
    
    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Hyperparameter Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Show top correlations with reward
    reward_corr = corr_matrix['reward'].drop('reward').sort_values(ascending=False, key=abs)
    
    print("\n✓ Top Hyperparameters Correlated with Reward:")
    print("="*70)
    for hp, corr in reward_corr.head(10).items():
        print(f" {hp:25s}: {corr:6.3f}")
    print("="*70)
    
    # Display comprehensive summary
    print("\n" + "="*70)
    print("DOA-DQN OPTIMIZATION SUMMARY FOR ATARI")
    print("="*70)
    print(f"\nEnvironment: {CONFIG['env_name']}")
    print(f"Total Evaluations: {doa_info['n_evals']}")
    print(f"DOA Population Size: {CONFIG['doa_pop_size']}")
    print(f"DOA Iterations: {CONFIG['doa_iterations']}")
    print(f"DQN Training Steps per Evaluation: {CONFIG['dqn_train_steps']}")
    
    print(f"\n{'Reward Statistics':^70}")
    print("-"*70)
    print(f" Best Reward: {best_reward:>10.2f}")
    print(f" Mean Reward: {np.mean(rewards):>10.2f}")
    print(f" Std Reward: {np.std(rewards):>10.2f}")
    print(f" Min Reward: {np.min(rewards):>10.2f}")
    print(f" Max Reward: {np.max(rewards):>10.2f}")
    print(f" Improvement: {best_reward - np.min(rewards):>10.2f}")
    
    print(f"\n{'SOTA Comparison (Atari)':^70}")
    print("-"*70)
    sota_dqn = SOTABaselinesAtari57.get_sota_baseline('DQN')
    sota_dqn_score = sota_dqn.get('mean_hns', sota_dqn.get('median_hns', 0))
    print(f" DQN (SOTA): {sota_dqn_score:.2f}")
    print(f" DOA-DQN: {best_reward:.2f}")
    print(f" Improvement: {best_reward - sota_dqn_score:+.2f}")
    
    print(f"\n{'Best Hyperparameters':^70}")
    print("-"*70)
    for key, value in sorted(best_hp.items()):
        if isinstance(value, float):
            print(f" {key:25s}: {value:>15.6f}")
        else:
            print(f" {key:25s}: {value:>15}")
    
    print("\n" + "="*70)
    print("Optimization completed successfully!")
    print("="*70)
