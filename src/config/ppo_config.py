from dataclasses import dataclass

@dataclass
class PPOConfig:
    exp_name: str = "ms_ai_ppo"
    learning_rate: float = 2.5e-4
    seed: int = 1
    total_timesteps: int = 2.5e6
    torch_deterministic: bool = False # torch.backends.cudnn.deterministic
    cuda: bool = False # use cuda
    track: bool = False # track training with wandb
    wandb_project: str = "minesweeper_ppo"
    capture_video: bool = True # capture video of agent playing
    
    # Algorithm specific arguments
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True # toggle learning rate annealing
    gae: bool = True # toggle generalized advantage estimation
    gamma: float = 0.99 # discount factor
    gae_lambda: float = 0.95 # gae lambda parameter
    num_mini_batch: int = 4 # number of mini batches
    update_epoches: int = 4 # number of epochs to update policy
    norm_adv: bool = True # normalize advantages
    clip_coef: float = 0.2 # clip parameter for PPO
    clip_value_loss: bool = True # clip value loss
    ent_coef: float = 0.01 # entropy coefficient
    vf_coef: float = 0.5 # value function coefficient
    max_grad_norm: float = 0.5 # max gradient norm
    target_kl: float = None # target kl divergence
    
    batch_size = int(num_envs * num_steps)
    mini_batch_size = int(batch_size // num_mini_batch)