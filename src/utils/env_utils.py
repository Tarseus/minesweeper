import gym
from env.minesweeper import MinesweeperEnv
from gym.wrappers import RecordVideo, RecordEpisodeStatistics

def make_env(env_in, seed, idx, capture_video=False, run_name=None):
    """Create a function that creates a new environment"""
    def thunk():
        assert isinstance(env_in, gym.Env)
        env = RecordEpisodeStatistics(env_in)
        
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
