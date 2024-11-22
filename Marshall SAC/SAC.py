from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from WildfireEnv import WildfireEnv
from WildfireCNN import WildfireCNN
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
import os
import torch

def plot_training_rewards(log_dir="sac_wildfire_logs"):
    x, y = ts2xy(load_results(log_dir), "timesteps")
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="Episode Rewards")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Time")
    plt.legend()
    plt.grid()
    plt.show()

log_dir = "sac_wildfire_logs"
os.makedirs(log_dir, exist_ok=True)

env = WildfireEnv()

env = Monitor(env, log_dir)

check_env(env)

policy_kwargs = dict(
    features_extractor_class=WildfireCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

if torch.cuda.is_available(): 
    model = SAC("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, buffer_size=50000, batch_size=64, device="cuda")
else:
    model = SAC("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, buffer_size=50000, batch_size=64)


model.learn(total_timesteps=5000)

model.save("sac_wildfire")

plot_training_rewards(log_dir="sac_wildfire_logs")

model = SAC.load("sac_wildfire")

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    print(f"Predicted Spread: x: {action[0]} y: {action[1]}, Reward: {reward}")