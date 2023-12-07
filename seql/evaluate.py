import torch
import robotic_warehouse
# import lbforaging
import gym

from agent import Agent
from wrappers import RecordEpisodeStatistics, TimeLimit

# import attack functions
from attack import fgsm, rand_noise, gaussian_noise, pgd

path = "/home/gwr/python_projects/Robust_Robotic_Warehouse/seac/results/unzip_models/rware-tiny-4ag-v1/u2000000" #"pretrained/rware-small-4ag"
env_name = "rware-tiny-4ag-v1"
time_limit = 500 # 25 for LBF
adv = "fgsm" # "fgsm", "pgd", "rand_noise", "gaussian_noise" and None
epsilon = 0.02 # <=0.02 is the appropriate perturbation size for fgsm and pgd, random noise and gaussian noise require larger perturbations
niters = 10 # for pgd

RUN_STEPS = 1500

env = gym.make(env_name)
env = TimeLimit(env, time_limit)
env = RecordEpisodeStatistics(env)

agents = Agent()

for agent in agents:
    agent.restore(path + f"/agent{agent.agent_id}")

obs = env.reset()

if not adv:
    for i in range(RUN_STEPS):
        obs = [torch.from_numpy(o) for o in obs]
        _, actions, _ , _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        # env.render()
        obs, _, done, info = env.step(actions)
        if all(done):
            obs = env.reset()
            print("--- Episode Finished ---")
            print(f"Episode rewards: {sum(info['episode_reward'])}")
            print(info)
            print(" --- ")
elif adv == "fgsm":
    for i in range(RUN_STEPS):
        obs = [torch.from_numpy(o) for o in obs]
        _, actions, _ , _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        adv_obs = fgsm(agents, epsilon, obs, actions, agents[0].optimizer)
        _, actions, _ , _ = zip(*[agent.model.act(adv_obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        # env.render()
        obs, _, done, info = env.step(actions)
        if all(done):
            obs = env.reset()
            print("--- Episode Finished ---")
            print(f"Episode rewards: {sum(info['episode_reward'])}")
            print(info)
            print(" --- ")
elif adv == "pgd":
    for i in range(RUN_STEPS):
        obs = [torch.from_numpy(o) for o in obs]
        _, actions, _ , _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        adv_obs = pgd(agents, epsilon, obs, actions, agents[0].optimizer, niters)
        _, actions, _ , _ = zip(*[agent.model.act(adv_obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        # env.render()
        obs, _, done, info = env.step(actions)
        if all(done):
            obs = env.reset()
            print("--- Episode Finished ---")
            print(f"Episode rewards: {sum(info['episode_reward'])}")
            print(info)
            print(" --- ")
elif adv == "rand_noise":
    for i in range(RUN_STEPS):
        obs = [torch.from_numpy(o) for o in obs]
        adv_obs = rand_noise(epsilon, obs)
        _, actions, _ , _ = zip(*[agent.model.act(adv_obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        # env.render()
        obs, _, done, info = env.step(actions)
        if all(done):
            obs = env.reset()
            print("--- Episode Finished ---")
            print(f"Episode rewards: {sum(info['episode_reward'])}")
            print(info)
            print(" --- ")
elif adv == "gaussian_noise":
    for i in range(RUN_STEPS):
        obs = [torch.from_numpy(o) for o in obs]
        adv_obs = gaussian_noise(epsilon, obs)
        _, actions, _ , _ = zip(*[agent.model.act(adv_obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        # env.render()
        obs, _, done, info = env.step(actions)
        if all(done):
            obs = env.reset()
            print("--- Episode Finished ---")
            print(f"Episode rewards: {sum(info['episode_reward'])}")
            print(info)
            print(" --- ")
else:
    print("Error: please specify a type of attack!")