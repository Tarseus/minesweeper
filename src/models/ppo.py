import sys
import os

# Add the src directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from config.ppo_config import PPOConfig
import numpy as np
import gym
import torch
import random
from env.minesweeper import MinesweeperEnv
import torch.nn as nn
from torch.distributions import Categorical
from utils.env_utils import make_env
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        h, w = envs.single_observation_space.shape
        num_channels = 11
        
        self.shared_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * h * w, 512),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, envs.single_action_space.n)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def get_value(self, x):
        b, h, w = x.shape
        x = x.long()
        x = F.one_hot(x, num_classes=11).permute(0, 4, 2, 3).float()
        x = self.shared_layers(x)
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None, action_mask=None):
        b, h, w = x.shape
        if torch.all(x == 10):
            center_x = w // 2
            center_y = h // 2
            action = center_x * h + center_y
        x = x.long()
        x = F.one_hot(x, num_classes=11).permute(0, 3, 1, 2).float()
        x = self.shared_layers(x)
        logits = self.actor(x)
        
        if action_mask is not None:
            action_mask = np.vstack(action_mask).astype(bool)
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
            logits[~mask] = float('-inf')
            
        probs = Categorical(logits=logits)
        if action is not None and not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.long, device=logits.device)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
if __name__ == "__main__":
    config = PPOConfig()
    run_name = f"{config.exp_name}_{config.seed}_{int(time.time())}"
    if config.track:
        import wandb
        wandb.init(
            project=config.wandb_project, 
            sync_tensorboard=True,
            config=vars(config),
            name=config.exp_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic
    
    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")
    
    env = MinesweeperEnv(
        width=8,
        height=8,
        num_mines=10,
    )
    envs = gym.vector.SyncVectorEnv(
        [make_env(env, config.seed + i, i, config.capture_video, run_name) for i in range(config.num_envs)]
    )
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only Discrete action spaces are supported"

    agent = Agent(envs).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    
    obs = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config.num_steps, config.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    
    global_step = 0
    start_time = time.time()
    obs1, info = envs.reset()
    next_obs = torch.Tensor(obs1).to(device)
    next_done = torch.zeros(config.num_envs).to(device)
    num_updates = config.total_timesteps // config.batch_size
    action_masks = info["action_mask"]
    
    for update in range(num_updates):
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = config.learning_rate * frac
            optimizer.param_groups[0]['lr'] = lrnow
        
        # rollout trajectory
        for step in range(0, config.num_steps):
            global_step += 1 * config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, 
                                                                       action_mask=action_masks)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            action_masks = info["action_mask"]
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if "final_info" in info:
                for item in info["final_info"]:
                    if item is not None:
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if config.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + config.gamma * nextnonterminal * next_return
                advantages = returns - values
            
            # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config.batch_size)
        clipfracs = []
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None:
                if approx_kl > config.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()