import torch
import robotic_warehouse
# import lbforaging
import gym

from a2c import A2C
from wrappers import RecordEpisodeStatistics, TimeLimit

# import attack functions
from attack import fgsm, rand_noise

path = "/home/gwr/python_projects/Robust_Robotic_Warehouse/seac/results/unzip_models/rware-tiny-4ag-v1/u2000000" #"pretrained/rware-small-4ag"
env_name = "rware-tiny-4ag-v1"
time_limit = 500 # 25 for LBF
adv = None # "fgsm", "pgd", "rand_noise", "gaussian_noise" and None
epsilon = 0.02 # <=0.02 is the appropriate perturbation size for fgsm

RUN_STEPS = 15000

env = gym.make(env_name)
env = TimeLimit(env, time_limit)
env = RecordEpisodeStatistics(env)

agents = [
    A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
    for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
]

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