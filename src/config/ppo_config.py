from dataclasses import dataclass, field

@dataclass
class PPOConfig:
    difficulty: str = "unknown"  # "beginner", "intermediate", "expert"
    width: int = field(init=False)
    height: int = field(init=False)
    num_mines: int = field(init=False)
    total_timesteps: int = field(init=False)
    num_steps: int = field(init=False)
    use_dfs: bool = True
    safe_center: bool = False # 中心点与其周围八个格子均不为雷
    first_click_safe: bool = True # 第一次点击的格子不为雷
    
    # General arguments
    learning_rate: float = 2.5e-4
    # learning_rate: float = 1e-6
    seed: int = 1
    torch_deterministic: bool = False # torch.backends.cudnn.deterministic
    cuda: bool = True # use cuda
    track: bool = True # track training with wandb
    wandb_project: str = "minesweeper_ppo"
    capture_video: bool = False # capture video of agent playing
    capture_video_freq: int = 1000
    save_freq: int = 1000 # save model every n updates
    phase: str = "train" # "train" or "test"
    use_full_board: bool = True # use full board as observation, otherwise use only the visible cells
    
    # Algorithm specific arguments
    num_envs: int = 8
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
    
    # Pretrain arguments
    use_pretrain: bool = False
    pretrain_model_path: str = "checkpoints/beginner_fullboard_1e8.pth"
    pretrain_total_timesteps: int = 8_192_000   # int(5e6)
    pretrain_update_epoches: int = 4
    test_model_path: str = "checkpoints/beginner_fullboard_2e8.pth"

    def get(self, key, default):
        return getattr(self, key) if hasattr(self, key) else default
    
    def __post_init__(self):
        assert self.difficulty in ["beginner", "intermediate", "expert"]
        if self.difficulty == "beginner":
            self.width = 9
            self.height = 9
            self.num_mines = 10
            self.total_timesteps = int(2e8)
            self.num_steps = 256
        elif self.difficulty == "intermediate":
            self.width = 16
            self.height = 16
            self.num_mines = 40
            self.total_timesteps = int(2e8)
            self.num_steps = 128
        elif self.difficulty == "expert":
            self.width = 30
            self.height = 16
            self.num_mines = 99
            self.total_timesteps = int(3e8)
            self.num_steps = 512
            
        self.batch_size = int(self.num_envs * self.num_steps)
        self.mini_batch_size = int(self.batch_size // self.num_mini_batch)
        self.exp_name: str = "ms_ai_ppo_" + self.difficulty