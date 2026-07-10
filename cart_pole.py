#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file train_cartpole_ppo.py
@brief 使用 Gymnasium 和 PyTorch 实现 PPO，训练 CartPole-v1 倒立摆平衡任务。

运行:
    python train_cartpole_ppo.py

依赖:
    pip install gymnasium torch numpy matplotlib
"""

import random
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    env_name: str = "CartPole-v1"
    seed: int = 42

    total_timesteps: int = 200_000
    rollout_steps: int = 2048

    gamma: float = 0.99
    gae_lambda: float = 0.95

    learning_rate: float = 3e-4
    update_epochs: int = 10
    minibatch_size: int = 64

    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    hidden_dim: int = 64
    eval_interval: int = 10_000
    eval_episodes: int = 5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "ppo_cartpole.pt"


def set_seed(seed: int) -> None:
    """@brief 设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ActorCritic(nn.Module):
    """
    @brief PPO Actor-Critic 网络。

    Actor:
        输出离散动作概率。

    Critic:
        输出状态价值 V(s)。
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        hidden = self.shared_net(obs)
        logits = self.actor(hidden)
        dist = Categorical(logits=logits)
        value = self.critic(hidden).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(self, obs: np.ndarray, device: str) -> Tuple[int, float, float]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        dist, value = self.forward(obs_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def greedy_action(self, obs: np.ndarray, device: str) -> int:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist, _ = self.forward(obs_tensor)
        return int(torch.argmax(dist.probs, dim=-1).item())

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, values = self.forward(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    @brief 计算 GAE advantage 和 return target。

    GAE:
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        A_t = delta_t + gamma * lambda * A_{t+1}
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        next_nonterminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * gae_lambda * next_nonterminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def evaluate_policy(
    env_name: str,
    agent: ActorCritic,
    config: PPOConfig,
) -> float:
    """@brief 用确定性动作评估当前策略。"""
    eval_returns = []

    for episode in range(config.eval_episodes):
        env = gym.make(env_name)
        obs, info = env.reset(seed=config.seed + 1000 + episode)

        episode_return = 0.0
        done = False

        while not done:
            action = agent.greedy_action(obs, config.device)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_return += reward

        env.close()
        eval_returns.append(episode_return)

    return float(np.mean(eval_returns))


def train() -> None:
    config = PPOConfig()
    set_seed(config.seed)

    env = gym.make(config.env_name)
    obs, info = env.reset(seed=config.seed)
    env.action_space.seed(config.seed)
    env.observation_space.seed(config.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
    ).to(config.device)

    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate)

    num_updates = config.total_timesteps // config.rollout_steps

    episode_returns = []
    current_episode_return = 0.0

    global_step = 0
    best_eval_return = -np.inf

    print("========== PPO CartPole-v1 ==========")
    print(f"device      : {config.device}")
    print(f"obs_dim     : {obs_dim}")
    print(f"action_dim  : {action_dim}")
    print(f"updates     : {num_updates}")
    print("=====================================")

    for update in range(1, num_updates + 1):
        obs_buffer = np.zeros((config.rollout_steps, obs_dim), dtype=np.float32)
        action_buffer = np.zeros(config.rollout_steps, dtype=np.int64)
        log_prob_buffer = np.zeros(config.rollout_steps, dtype=np.float32)
        reward_buffer = np.zeros(config.rollout_steps, dtype=np.float32)
        done_buffer = np.zeros(config.rollout_steps, dtype=np.float32)
        value_buffer = np.zeros(config.rollout_steps, dtype=np.float32)

        for step in range(config.rollout_steps):
            global_step += 1

            action, log_prob, value = agent.act(obs, config.device)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs_buffer[step] = obs
            action_buffer[step] = action
            log_prob_buffer[step] = log_prob
            reward_buffer[step] = reward
            done_buffer[step] = float(done)
            value_buffer[step] = value

            current_episode_return += reward

            obs = next_obs

            if done:
                episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                obs, info = env.reset()

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=config.device).unsqueeze(0)
            _, last_value_tensor = agent.forward(obs_tensor)
            last_value = float(last_value_tensor.item())

        advantages, returns = compute_gae(
            rewards=reward_buffer,
            dones=done_buffer,
            values=value_buffer,
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_tensor = torch.as_tensor(obs_buffer, dtype=torch.float32, device=config.device)
        action_tensor = torch.as_tensor(action_buffer, dtype=torch.long, device=config.device)
        old_log_prob_tensor = torch.as_tensor(log_prob_buffer, dtype=torch.float32, device=config.device)
        advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=config.device)
        return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=config.device)

        batch_size = config.rollout_steps
        indices = np.arange(batch_size)

        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []

        for epoch in range(config.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_idx = indices[start:end]

                mb_obs = obs_tensor[mb_idx]
                mb_actions = action_tensor[mb_idx]
                mb_old_log_probs = old_log_prob_tensor[mb_idx]
                mb_advantages = advantage_tensor[mb_idx]
                mb_returns = return_tensor[mb_idx]

                new_log_probs, entropy, new_values = agent.evaluate_actions(
                    mb_obs,
                    mb_actions,
                )

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                unclipped_policy_loss = -mb_advantages * ratio
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - config.clip_coef,
                    1.0 + config.clip_coef,
                )
                clipped_policy_loss = -mb_advantages * clipped_ratio

                policy_loss = torch.max(
                    unclipped_policy_loss,
                    clipped_policy_loss,
                ).mean()

                value_loss = 0.5 * torch.mean((new_values - mb_returns) ** 2)
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = torch.mean((ratio - 1.0) - log_ratio).item()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_losses.append(float(entropy_loss.item()))
                approx_kls.append(float(approx_kl))

        if global_step % config.eval_interval < config.rollout_steps:
            eval_return = evaluate_policy(config.env_name, agent, config)

            if eval_return > best_eval_return:
                best_eval_return = eval_return
                torch.save(agent.state_dict(), config.save_path)

            recent_train_return = (
                np.mean(episode_returns[-10:]) if len(episode_returns) > 0 else 0.0
            )

            print(
                f"step={global_step:7d} "
                f"update={update:4d} "
                f"train_return={recent_train_return:8.2f} "
                f"eval_return={eval_return:8.2f} "
                f"best_eval={best_eval_return:8.2f} "
                f"pi_loss={np.mean(policy_losses): .4f} "
                f"v_loss={np.mean(value_losses): .4f} "
                f"entropy={np.mean(entropy_losses): .4f} "
                f"kl={np.mean(approx_kls): .6f}"
            )

    env.close()

    print(f"Training finished. Best model saved to: {config.save_path}")

    if len(episode_returns) > 0:
        plt.figure()
        plt.plot(episode_returns)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("PPO on CartPole-v1")
        plt.grid(True)
        plt.show()


def play() -> None:
    """@brief 加载训练好的策略并可视化运行。"""
    config = PPOConfig()

    env = gym.make(config.env_name, render_mode="human")
    obs, info = env.reset(seed=config.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
    ).to(config.device)

    agent.load_state_dict(torch.load(config.save_path, map_location=config.device))
    agent.eval()

    episode_return = 0.0
    done = False

    while not done:
        action = agent.greedy_action(obs, config.device)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_return += reward

    print(f"play episode return: {episode_return}")
    env.close()


def test():
    env = gym.make('CartPole-v1', render_mode="human")
    obs, info = env.reset(seed=20)
    done = False
    episode_return = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_return += reward

    print(f"play episode return: {episode_return}")
    env.close()

if __name__ == "__main__":
    # train()

    # 训练结束后想看效果，可以取消下面这一行注释。
    test()