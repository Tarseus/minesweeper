import sys
import os

# Add the src directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from config.ppo_config import PPOConfig
import numpy as np
import gymnasium as gym
import torch
import random
from env.minesweeper import MinesweeperEnv
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
from models.ppo import Agent
from models.logic import LogicSolver
from utils.env_utils import make_env
import time

if __name__ == "__main__":
    config = PPOConfig()
    run_name = f"pretrain_{config.exp_name}_{config.seed}_{time.strftime('%d/%m/%Y_%H-%M-%S')}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    if config.track:
        import wandb
        wandb.init(
            project=config.wandb_project, 
            sync_tensorboard=True,
            config=vars(config),
            name=config.exp_name + "_pretrain",
            monitor_gym=True,
            save_code=True,
        )
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(config, config.seed + i, i, False, run_name) for i in range(config.num_envs)]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only Discrete action spaces are supported"

    agent = Agent(envs).to(device)
    

    pretrain_optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    num_updates = config.pretrain_total_timesteps // config.batch_size
    pretrain_progress_bar = tqdm(range(num_updates), dynamic_ncols=True)

    for update in pretrain_progress_bar:
        obs1, info = envs.reset()
        logic_agent = LogicSolver(envs)
        obs = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((config.num_steps, config.num_envs) + envs.single_action_space.shape).to(device)
        rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
        dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
        next_obs = torch.Tensor(obs1).to(device)
        next_done = torch.zeros(config.num_envs).to(device)

        for step in range(0, config.num_steps):
            obs[step] = next_obs
            dones[step] = next_done

            action = logic_agent.get_actions(next_obs)
            action = torch.tensor(action).to(device).view(-1)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            for i in range(config.num_envs):
                if "final_info" in infos:
                    if infos["final_info"][i] is not None:
                        logic_agent.moves_made[i].clear()
                        logic_agent.mines[i].clear()
                        logic_agent.safes[i].clear()
                        logic_agent.knowledge[i].clear()
            action_masks = info["action_mask"]
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        returns = torch.zeros((config.num_steps, config.num_envs)).to(device)
        for step in reversed(range(config.num_steps)):
            if step == config.num_steps - 1:
                next_return = agent.get_value(next_obs).view(-1)
                next_nonterminal = 1.0 - next_done
            else:
                next_return = returns[step + 1]
                next_nonterminal = 1.0 - dones[step + 1]
            returns[step] = rewards[step] + config.gamma * next_nonterminal * next_return

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)

        pretrain_optimizer.zero_grad()
        _, logprob, _, value = agent.get_action_and_value(b_obs, b_actions)
        value = value.view(-1)
        pretrain_value_loss = F.mse_loss(value, b_returns)
        pretrain_policy_loss = -logprob.mean()
        pretrain_loss = pretrain_value_loss + pretrain_policy_loss
        pretrain_loss.backward()
        pretrain_optimizer.step()

        writer.add_scalar("losses/pretrain_value_loss", pretrain_value_loss.item(), update)
        writer.add_scalar("losses/pretrain_policy_loss", pretrain_policy_loss.item(), update)
        writer.add_scalar("losses/pretrain_loss", pretrain_loss.item(), update)
        pretrain_progress_bar.set_postfix({
            'p_loss': f'{pretrain_policy_loss.item():.3f}',
            'v_loss': f'{pretrain_value_loss.item():.3f}',
            'total_loss': f'{pretrain_loss.item():.3f}'
        })
        if config.track:
            wandb.log({
                "train/value_loss": pretrain_value_loss.item(),
                "train/policy_loss": pretrain_policy_loss.item(),
                "train/total_loss": pretrain_loss.item(),
            }, step=update)

    envs.close()
    writer.close()

    torch.save(agent.state_dict(), config.pretrain_model_path + "_" + {config.difficulty} + ".pth")