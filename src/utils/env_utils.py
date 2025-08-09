import gymnasium as gym
from src.env.minesweeper import MinesweeperEnv
from gym.wrappers import RecordVideo, RecordEpisodeStatistics

def make_env(env_config, seed, idx, capture_video=False, run_name=None):
    """Create a function that creates a new environment"""
    def thunk():
        env = MinesweeperEnv(env_config)
        env = RecordEpisodeStatistics(env)
        
        if capture_video:
            if idx == 0:
                if run_name:
                    videos_dir = f"videos/{run_name}"
                else:
                    videos_dir = "videos"
                env = gym.wrappers.RecordVideo(
                    env, 
                    videos_dir,
                    episode_trigger=lambda x: x % 100 == 0
                )
        
        env.seed(seed)
        return env
        
    return thunk
