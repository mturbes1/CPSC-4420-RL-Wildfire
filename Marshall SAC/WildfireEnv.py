import gymnasium as gym
import numpy as np
import os
import cv2
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt

class WildfireEnv(gym.Env):
    def __init__(self):
        super(WildfireEnv, self).__init__()
        self.frames = self.load_frames()
        self.current_step = 0

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
    
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def load_frames(self):
        base_dir_RGB = "CPSC 6420 Flame 2 Research/Segmented Frames/RGB"
        base_dir_IR = "CPSC 6420 Flame 2 Research/Segmented Frames/IR"

        frame_pairs = []
        for video_index in range(1, 8):
            rgb_video_dir = os.path.join(base_dir_RGB, f"Video {video_index}")
            ir_video_dir = os.path.join(base_dir_IR, f"Video {video_index}")

            rgb_frames = sorted(os.listdir(rgb_video_dir))
            ir_frames = sorted(os.listdir(ir_video_dir))

            for rgb_frame_name, ir_frame_name in zip(rgb_frames, ir_frames):
                rgb_frame_path = os.path.join(rgb_video_dir, rgb_frame_name)
                ir_frame_path = os.path.join(ir_video_dir, ir_frame_name)

                rgb_frame = cv2.imread(rgb_frame_path)   
                ir_frame = cv2.imread(ir_frame_path)
                rgb_frame = cv2.resize(rgb_frame, (84, 84))
                ir_frame = cv2.resize(ir_frame, (84, 84))    

                frame_pairs.append((rgb_frame, ir_frame))
        return frame_pairs

    def reset(self, seed=None, options=None):
        print(f"Reset called, current_step={self.current_step}")  
        self.current_step = 0
        _, ir_frame = self.frames[self.current_step]
        ir_frame = np.array(ir_frame)
        
        return ir_frame, {}

    def calculate_reward(self, action, ir_frame, next_ir_frame, rgb_frame, next_rgb_frame):
        actual_vector = self.estimate_actual_direction(ir_frame, next_ir_frame, rgb_frame, next_rgb_frame)

        norm_action = action / (np.linalg.norm(action) + 1e-6)
        norm_actual = actual_vector / (np.linalg.norm(actual_vector) + 1e-6)

        reward = np.dot(norm_action, norm_actual)
        return reward
    
    def step(self, action):
        action = action / (np.linalg.norm(action) + 1e-6)
                           
        next_step = (self.current_step + 1) % len(self.frames)
        rgb_frame, ir_frame = self.frames[self.current_step]
        next_rgb_frame, next_ir_frame = self.frames[next_step]

        reward = self.calculate_reward(action, ir_frame, next_ir_frame, rgb_frame, next_rgb_frame)
        
        self.current_step = next_step
        done = self.current_step == 0 
        
        info = {"current_step": self.current_step}
        return ir_frame, reward, done, False, info

    def visualize_fire_movement(self, prev_rgb_frame, next_rgb_frame, direction_vector, previous_center):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(prev_rgb_frame, cv2.COLOR_BGR2RGB))
        plt.title("Previous Frame")
        plt.axis("off")
        
        next_frame_with_overlay = next_rgb_frame.copy()
        if previous_center is not None:
            start_point = (int(previous_center[1]), int(previous_center[0]))
            end_point = (
                int(previous_center[1] + direction_vector[1] * 10),
                int(previous_center[0] + direction_vector[0] * 10)
            )
            cv2.arrowedLine(
                next_frame_with_overlay,
                start_point,
                end_point,
                color=(0, 255, 0),
                thickness=1,
                tipLength=0.3
            )
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(next_frame_with_overlay, cv2.COLOR_BGR2RGB))
        plt.title("Next Frame with Movement Vector")
        plt.axis("off")
        
        plt.show()

    def estimate_actual_direction(self, ir_frame, next_ir_frame, rgb_frame, next_rgb_frame):
        #Turn to grayscale
        if len(ir_frame.shape) > 2:
            ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
        if len(next_ir_frame.shape) > 2:
            next_ir_frame = cv2.cvtColor(next_ir_frame, cv2.COLOR_BGR2GRAY)

        if ir_frame.shape != next_ir_frame.shape:
            next_ir_frame = cv2.resize(next_ir_frame, (ir_frame.shape[1], ir_frame.shape[0]))
        
        threshold = ir_frame.mean() + 0.5 * ir_frame.std()

        fire_mask = (ir_frame > threshold).astype(np.uint8)
        next_fire_mask = (next_ir_frame > threshold).astype(np.uint8)

        previous_center = center_of_mass(fire_mask)
        next_center = center_of_mass(next_fire_mask)
        
        #If no fire
        if previous_center is None or next_center is None:
            return np.array([0.0, 0.0])

        com_direction_vector = np.array(next_center) - np.array(previous_center)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev=ir_frame,
            next=next_ir_frame,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        
        flow_horizontal = flow[..., 0] * fire_mask 
        flow_vertical = flow[..., 1] * fire_mask  
        
        num_active_pixels = np.sum(fire_mask)
        if num_active_pixels > 0:
            avg_flow_vector = np.array([np.sum(flow_horizontal) / num_active_pixels, np.sum(flow_vertical) / num_active_pixels])
        else:
            avg_flow_vector = np.array([0.0, 0.0])
        
        #Find average of direction vector and flow vector
        direction_vector = 0.5 * com_direction_vector + 0.5 * avg_flow_vector

        #self.visualize_fire_movement(rgb_frame, next_rgb_frame, direction_vector, previous_center)

        return direction_vector