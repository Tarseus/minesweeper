# filepath: /e:/Small_projects/minesweeper/src/utils/env_utils.py
import gym
from env.minesweeper import MinesweeperEnv
from gym.wrappers import RecordEpisodeStatistics
import pygame
from moviepy.editor import ImageSequenceClip
import os

def make_env(env_config, seed, idx, capture_video=False, run_name=None):
    """Create a function that creates a new environment"""
    def thunk():
        env = MinesweeperEnv(env_config)
        env = RecordEpisodeStatistics(env)
        video_recorder = None
        frames = []  # List to store frames

        if capture_video:
            if idx == 0:
                if run_name:
                    videos_dir = f"videos/{run_name}"
                else:
                    videos_dir = "videos"
                os.makedirs(videos_dir, exist_ok=True)
                video_path = os.path.join(videos_dir, f"minesweeper-{run_name or 'default'}.mp4")
                
                def record_frame():
                    try:
                        # Get the Pygame display surface
                        frame = pygame.surfarray.array3d(env.render(mode="human"))
                        frame = frame.swapaxes(0, 1)
                        frames.append(frame)
                    except Exception as e:
                        print(f"Error capturing frame: {e}")

                env.record_frame = record_frame  # Attach the function to the environment

        env.seed(seed)

        try:
            yield env
        finally:
            if frames:
                try:
                    # Create a video clip from the frames
                    clip = ImageSequenceClip(frames, fps=30)  # Adjust fps as needed
                    clip.write_videofile(video_path, codec='libx264')
                    print(f"Video saved to {video_path}")
                except Exception as e:
                    print(f"Error saving video: {e}")

    gen = thunk()
    env = next(gen)
    return lambda: next(gen, env)