import gymnasium as gym
import numpy as np
import cv2
from datetime import datetime
import os

class VideoRecoder:
    def __init__(
        self,
        videos_dir: str,
        fps: int = 1,
        name_prefix: str = "ms-video",
        if_save_frames: bool = False,
    ):
        self.stored_frames = []
        self.videos_dir = videos_dir
        self.fps = fps
        self.name_prefix = name_prefix
        self.save_frames = if_save_frames
        self.frames_dir = os.path.join(videos_dir, f"{name_prefix}-frames")
        if if_save_frames:
            os.makedirs(self.frames_dir, exist_ok=True)
        self.frame_count = 0
        
    def record_frame(self, frame):
        self.stored_frames.append(frame)
        if self.save_frames:
            self.save_frame(frame)
            self.frame_count += 1
            
    def save_frame(self, frame):
        frame_name = f"{self.frame_count}.png"
        frame_path = os.path.join(self.frames_dir, frame_name)
        cv2.imwrite(frame_path, frame)
    
    def save_video(self):
        if len(self.stored_frames) == 0:
            return
        
        height, width, _ = self.stored_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_name = f"{self.name_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4"
        video_name = f"{self.name_prefix}-1.mp4"
        video_path = os.path.join(self.videos_dir, video_name)
        out = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
        try:
            for frame in self.stored_frames:
                out.write(frame)
        except Exception as e:
            print(f"保存视频时出错: {str(e)}")
        finally:
            out.release()
            self.stored_frames = []
        
class VideoRecorderWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        videos_dir: str,
        fps: int = 1,
        name_prefix: str = "ms-video",
        if_save_frames: bool = False,
    ):
        super().__init__(env)
        
        self.videos_dir = videos_dir
        os.makedirs(videos_dir, exist_ok=True)
        
        self.fps = fps
        self.name_prefix = name_prefix
        
        self.recording = False
        self.recorded_frames = 0
        self.recorder = VideoRecoder(videos_dir, fps, name_prefix, if_save_frames)
        
    def reset(self, **kwargs):
        self.recorded_frames = 0
        self.recording = True
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, is_win, info = self.env.step(action)
        
        if self.recording:
            frame = self.env.render(mode='pygame')
            self.recorder.record_frame(frame)
            self.recorded_frames += 1
            if done:
                frame = self.env.render(mode='pygame')
                self.recorder.record_frame(frame)
                self.recorded_frames += 1
            
        return obs, reward, done, is_win, info
    
    def close(self):
        if self.recording:
            self.recorder.save_video()
        self.env.close()