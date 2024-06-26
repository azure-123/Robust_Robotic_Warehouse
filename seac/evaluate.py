import torch
import robotic_warehouse
# import lbforaging
import gym

from a2c import A2C
from wrappers import RecordEpisodeStatistics, TimeLimit
import numpy as np

# import attack functions
from attack import fgsm, rand_noise, gaussian_noise, pgd, tar_attack

path = "/home/gwr/python_projects/Robust_Robotic_Warehouse/seac/results/unzip_models/rware-tiny-4ag-v1_pgd_0-02/u1000000" #"pretrained/rware-small-4ag"
adv_path = ""
env_name = "rware-tiny-4ag-v1"
time_limit = 500 # 25 for LBF
adv = "fgsm" # "fgsm", "pgd", "rand_noise", "gaussian_noise" and None
epsilon = 0.04 # <=0.02 is the appropriate perturbation size for fgsm and pgd, random noise and gaussian noise require larger perturbations
niters = 10 # for pgd

RUN_STEPS = time_limit * 20

env = gym.make(env_name)
env = TimeLimit(env, time_limit)
env = RecordEpisodeStatistics(env)

agents = [
    A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
    for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
]

for agent in agents:
    agent.restore(path + f"/agent{agent.agent_id}")

if adv == "adv_tar":
    adv_agents = [
    A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
    for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
]
    for adv_agent in adv_agents:
        adv_agent.restore(adv_path + f"/agent{adv_agent.agent_id}")

obs = env.reset()

if not adv:
    reward_list = []
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
            reward_list.append(sum(info['episode_reward']))
            print(info)
            print(" --- ")
    print(f"Reward mean: {np.mean(np.array(reward_list))}")
    print(f"Reward std: {np.std(np.array(reward_list))}")
elif adv == "fgsm":
    reward_list = []
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
            reward_list.append(sum(info['episode_reward']))
            print(info)
            print(" --- ")
    print(f"Reward mean: {np.mean(np.array(reward_list))}")
    print(f"Reward std: {np.std(np.array(reward_list))}")
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
            print(f"Reward std: {np.std(np.array(info['episode_reward']))}")
            print(info)
            print(" --- ")
elif adv == "rand_noise":
    reward_list = []
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
            reward_list.append(sum(info['episode_reward']))
            print(info)
            print(" --- ")
    print(f"Reward mean: {np.mean(np.array(reward_list))}")
    print(f"Reward std: {np.std(np.array(reward_list))}")
elif adv == "gaussian_noise":
    reward_list = []
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
            reward_list.append(sum(info['episode_reward']))
            print(info)
            print(" --- ")
    print(f"Reward mean: {np.mean(np.array(reward_list))}")
    print(f"Reward std: {np.std(np.array(reward_list))}")
elif adv == "adv_tar":
    for i in range(RUN_STEPS):
        obs = [torch.from_numpy(o) for o in obs]
        _, actions, _ , _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        _, tar_actions, _ , _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        adv_obs = tar_attack(agents, epsilon, obs, actions, tar_actions, adv_agents[0].optimizer)
        _, actions, _ , _ = zip(*[agent.model.act(adv_obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        # env.render()
        obs, _, done, info = env.step(actions)
        if all(done):
            obs = env.reset()
            print("--- Episode Finished ---")
            print(f"Episode rewards: {sum(info['episode_reward'])}")
            print(f"Reward std: {np.std(np.array(info['episode_reward']))}")
            print(info)
            print(" --- ")
else:
    print("Error: please specify a type of attack!")