import glob
import logging
import os
import shutil
import time
from collections import deque
from os import path
from pathlib import Path
import random

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import (  # noqa
    FileStorageObserver,
    MongoObserver,
    QueuedMongoObserver,
    QueueObserver,
)
from torch.utils.tensorboard import SummaryWriter

import utils
from a2c import A2C, algorithm
from envs import make_vec_envs
from wrappers import RecordEpisodeStatistics, SquashDones
from model import Policy

import robotic_warehouse # noqa
# import lbforaging # noqa
# attack
from attack import tar_attack, fgsm, pgd, rand_noise, gaussian_noise, atla
from mappo_attack.mappo import MAPPO

ex = Experiment(ingredients=[algorithm])
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."
ex.observers.append(FileStorageObserver("./results/sacred"))

logging.basicConfig(
    level=logging.INFO,
    format="(%(process)d) [%(levelname).1s] - (%(asctime)s) - %(name)s >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
)


@ex.config
def config():
    env_name = None
    time_limit = None
    wrappers = (
        RecordEpisodeStatistics,
        SquashDones,
    )
    dummy_vecenv = False

    num_env_steps = 40000000

    eval_dir = "./results/video/{id}"
    loss_dir = "./results/loss/{id}"
    save_dir = "./results/trained_models/{id}"

    log_interval = 2000
    save_interval = int(1e6)
    eval_interval = 4000000000 # int(1e6)
    episodes_per_eval = 8

    # attack config
    adv = "pgd" #"fgsm", "pgd", "rand_noise", "gaussian_noise", "atla", "adv_tar" and None
    epsilon_ball = 0.02

for conf in glob.glob("configs/*.yaml"):
    name = f"{Path(conf).stem}"
    ex.add_named_config(name, conf)

def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    for key in keys:
        mean = np.mean([np.array(d[key]).sum() for d in info if key in d])
        new_info[key] = mean
    return new_info


@ex.capture
def evaluate(
    agents,
    monitor_dir,
    episodes_per_eval,
    env_name,
    seed,
    wrappers,
    dummy_vecenv,
    time_limit,
    algorithm,
    _log,
):
    device = algorithm["device"]

    eval_envs = make_vec_envs(
        env_name,
        seed,
        dummy_vecenv,
        episodes_per_eval,
        time_limit,
        wrappers,
        device,
        # monitor_dir=monitor_dir,
    )
    # envs = make_vec_envs(
    #     env_name,
    #     seed,
    #     dummy_vecenv,
    #     algorithm["num_processes"],
    #     time_limit,
    #     wrappers,
    #     algorithm["device"],
    # )

    n_obs = eval_envs.reset()
    n_recurrent_hidden_states = [
        torch.zeros(
            episodes_per_eval, agent.model.recurrent_hidden_state_size, device=device
        )
        for agent in agents
    ]
    masks = torch.zeros(episodes_per_eval, 1, device=device)

    all_infos = []

    while len(all_infos) < episodes_per_eval:
        with torch.no_grad():
            _, n_action, _, n_recurrent_hidden_states = zip(
                *[
                    agent.model.act(
                        n_obs[agent.agent_id], recurrent_hidden_states, masks
                    )
                    for agent, recurrent_hidden_states in zip(
                        agents, n_recurrent_hidden_states
                    )
                ]
            )

        # Obser reward and next obs
        n_obs, _, done, infos = eval_envs.step(n_action)

        n_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )
        all_infos.extend([i for i in infos if i])

    eval_envs.close()
    info = _squash_info(all_infos)
    print(info)
    # _log.info(
    #     f"Evaluation using {len(all_infos)} episodes: mean reward {info['episode_reward']:.5f}\n"
    # )


@ex.automain
def main(
    _run,
    _log,
    num_env_steps,
    env_name,
    seed,
    algorithm,
    dummy_vecenv,
    time_limit,
    wrappers,
    save_dir,
    eval_dir,
    loss_dir,
    log_interval,
    save_interval,
    eval_interval,
    adv,
    epsilon_ball
):

    # random.seed(0)
    if loss_dir:
        loss_dir = path.expanduser(loss_dir.format(id=str(_run._id)))
        utils.cleanup_log_dir(loss_dir)
        writer = SummaryWriter(loss_dir)
    else:
        writer = None

    eval_dir = path.expanduser(eval_dir.format(id=str(_run._id)))
    save_dir = path.expanduser(save_dir.format(id=str(_run._id)))

    utils.cleanup_log_dir(eval_dir)
    utils.cleanup_log_dir(save_dir)

    torch.set_num_threads(1)
    envs = make_vec_envs(
        env_name,
        seed,
        dummy_vecenv,
        algorithm["num_processes"],
        time_limit,
        wrappers,
        algorithm["device"],
    )

    # adv = None
    agents = [
        A2C(i, osp, asp)
        for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
    ]

    if adv == "adv_tar":
        adv_agents = [
            A2C(i, osp, asp)
            for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
        ]
    elif adv == "atla":
        adv_agents = MAPPO(envs, envs.observation_space, envs.observation_space)

    obs = envs.reset()

    if adv == "adv_tar":
    # calculate the actions before purturbing
        clean_path = "/home/gwr/python_projects/Robust_Robotic_Warehouse/seac/results/unzip_models/rware-tiny-2ag-v1/u3000000" #"pretrained/rware-small-4ag"
        for agent in agents:
            agent.restore(clean_path + f"/agent{agent.agent_id}")
        with torch.no_grad():
            n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                            *[
                                agent.model.act(
                                    obs[agent.agent_id],
                                    agent.storage.recurrent_hidden_states[0],
                                    agent.storage.masks[0],
                                )
                                for agent in agents
                            ]
                        )
        adv_obs = obs
            
        #     adv_value, tar_action, adv_action_log_prob, adv_recurrent_hidden_states = zip(
        #                     *[
        #                         agent.model.act(
        #                             obs[agent.agent_id],
        #                             agent.storage.recurrent_hidden_states[0],
        #                             agent.storage.masks[0],
        #                         )
        #                         for agent in adv_agents
        #                     ]
        #                 )
        
        # adv_obs = tar_attack(agents, epsilon_ball, obs, n_action, tar_action, adv_agents[0].optimizer)

        
        for i in range(len(obs)):
            agents[i].storage.obs[0].copy_(adv_obs[i])
            agents[i].storage.to(algorithm["device"])
            # adv_agents[i].storage.obs[0].copy_(obs[i])
            # adv_agents[i].storage.to(algorithm["device"])

        start = time.time()
        num_updates = (
            int(num_env_steps) // algorithm["num_steps"] // algorithm["num_processes"]
        )

        all_infos = deque(maxlen=10)

        for j in range(1, num_updates + 1):

            for step in range(algorithm["num_steps"]):
                # Sample actions
                with torch.no_grad():
                    n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                        *[
                            agent.model.act(
                                agent.storage.obs[step],
                                agent.storage.recurrent_hidden_states[step],
                                agent.storage.masks[step],
                            )
                            for agent in agents
                        ]
                    )
                # Obser reward and next obs
                obs, reward, done, infos = envs.step(n_action) # n_action is the perturbed action
                if j > log_interval:
                    with torch.no_grad():
                        _, temp_action, _, _ = zip(
                            *[
                                agent.model.act(
                                    obs[agent.agent_id],
                                    agent.storage.recurrent_hidden_states[step],
                                    agent.storage.masks[step],
                                )
                                for agent in agents
                            ]
                        )
                        adv_value, tar_action, adv_action_log_prob, adv_recurrent_hidden_states = zip(
                                *[
                                    agent.model.act(
                                        obs[agent.agent_id],
                                        agent.storage.recurrent_hidden_states[step],
                                        agent.storage.masks[step],
                                    )
                                    for agent in adv_agents
                                ]
                            )
                if j > log_interval:
                    adv_obs = tar_attack(agents, epsilon_ball, obs, temp_action, tar_action, adv_agents[0].optimizer)
                else:
                    adv_obs = obs

                # envs.envs[0].render()

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

                bad_masks = torch.FloatTensor(
                    [
                        [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                        for info in infos
                    ]
                )
                for i in range(len(agents)):
                    agents[i].storage.insert(
                        adv_obs[i],
                        n_recurrent_hidden_states[i],
                        n_action[i],
                        n_action_log_prob[i],
                        n_value[i],
                        reward[:, i].unsqueeze(1),
                        masks,
                        bad_masks,
                    )
                if j > log_interval:
                    for i in range(len(adv_agents)):
                        adv_agents[i].storage.insert(
                            obs[i],
                            adv_recurrent_hidden_states[i],
                            tar_action[i],
                            adv_action_log_prob[i],
                            adv_value[i],
                            -reward[:, i].unsqueeze(1),
                            masks,
                            bad_masks,
                        )

                for info in infos:
                    if info:
                        all_infos.append(info)

            # value_loss, action_loss, dist_entropy = agent.update(rollouts)
            for agent in agents:
                agent.compute_returns()
            if j > log_interval:
                for adv_agent in adv_agents:
                    adv_agent.compute_returns()

            for agent in agents:
                loss = agent.update([a.storage for a in agents])
                for k, v in loss.items():
                    if writer:
                        writer.add_scalar(f"agent{agent.agent_id}/{k}", v, j)
            if j > log_interval:
                for adv_agent in adv_agents:
                    loss = adv_agent.update([a.storage for a in adv_agents])
                    for k, v in loss.items():
                        if writer:
                            writer.add_scalar(f"adv_agent{adv_agent.agent_id}/{k}", v, j)

            for agent in agents:
                agent.storage.after_update()
            if j > log_interval:
                for adv_agent in adv_agents:
                    adv_agent.storage.after_update()

            if j % log_interval == 0 and len(all_infos) > 1:
                squashed = _squash_info(all_infos)

                total_num_steps = (
                    (j + 1) * algorithm["num_processes"] * algorithm["num_steps"]
                )
                end = time.time()
                _log.info(
                    f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}"
                )
                _log.info(
                    f"Last {len(all_infos)} training episodes mean reward {squashed['episode_reward'].sum():.3f}"
                )

                for k, v in squashed.items():
                    _run.log_scalar(k, v, j)
                all_infos.clear()

            if save_interval is not None and (
                j > 0 and j % save_interval == 0 or j == num_updates
            ):
                cur_save_dir = path.join(save_dir, f"u{j}")
                for agent in agents:
                    save_at = path.join(cur_save_dir, f"agent{agent.agent_id}")
                    os.makedirs(save_at, exist_ok=True)
                    agent.save(save_at)
                if j > log_interval:
                    for adv_agent in agents:
                        save_at = path.join(cur_save_dir, f"adv_agent{adv_agent.agent_id}")
                        os.makedirs(save_at, exist_ok=True)
                        adv_agent.save(save_at)
                archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir, f"u{j}")
                shutil.rmtree(cur_save_dir)
                _run.add_artifact(archive_name)

            if eval_interval is not None and (
                j > 0 and j % eval_interval == 0 or j == num_updates
            ):
                evaluate(
                    agents, os.path.join(eval_dir, f"u{j}"),
                )
                videos = glob.glob(os.path.join(eval_dir, f"u{j}") + "/*.mp4")
                for i, v in enumerate(videos):
                    _run.add_artifact(v, f"u{j}.{i}.mp4")
    if adv == "atla":
    # calculate the actions before purturbing
        clean_path = "/home/gwr/python_projects/Robust_Robotic_Warehouse/seac/results/unzip_models/rware-tiny-4ag-v1/u2000000" #"pretrained/rware-small-4ag"
        for agent in agents:
            agent.restore(clean_path + f"/agent{agent.agent_id}")
        adv_obs, perturbations = atla(adv_agents, epsilon_ball, obs)
        
        for i in range(len(obs)):
            agents[i].storage.obs[0].copy_(adv_obs[i])
            agents[i].storage.to(algorithm["device"])
            adv_agents.storage[i].obs[0].copy_(obs[i])
            adv_agents.storage[i].to(algorithm["device"])
            # adv_agents[i].storage.obs[0].copy_(obs[i])
            # adv_agents[i].storage.to(algorithm["device"])
        

        start = time.time()
        num_updates = (
            int(num_env_steps) // algorithm["num_steps"] // algorithm["num_processes"]
        )

        all_infos = deque(maxlen=10)

        for j in range(1, num_updates + 1):

            for step in range(algorithm["num_steps"]):
                # Sample actions
                with torch.no_grad():
                    n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                        *[
                            agent.model.act(
                                agent.storage.obs[step],
                                agent.storage.recurrent_hidden_states[step],
                                agent.storage.masks[step],
                            )
                            for agent in agents
                        ]
                    )
                # Obser reward and next obs
                obs, reward, done, infos = envs.step(n_action) # n_action is the perturbed action
                adv_obs, perturbations = atla(adv_agents, epsilon_ball, obs)

                # envs.envs[0].render()

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

                bad_masks = torch.FloatTensor(
                    [
                        [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                        for info in infos
                    ]
                )
                for i in range(len(agents)):
                    agents[i].storage.insert(
                        adv_obs[i],
                        n_recurrent_hidden_states[i],
                        n_action[i],
                        n_action_log_prob[i],
                        n_value[i],
                        reward[:, i].unsqueeze(1),
                        masks,
                        bad_masks,
                    )
                    adv_agents.storage[i].insert(
                        obs[i],
                        # n_recurrent_hidden_states[i],
                        perturbations[i],
                        # n_action_log_prob[i],
                        # n_value[i],
                        -reward[:, i].unsqueeze(1),
                        masks,
                        bad_masks,

                    )

                # save the pertubation, i.e. the actions from the adversary
                # adv_agents.memory.push(obs, adv_obs, -reward) # push as lists

                for info in infos:
                    if info:
                        all_infos.append(info)

            # value_loss, action_loss, dist_entropy = agent.update(rollouts)
            for agent in agents:
                agent.compute_returns()

            for agent in agents:
                loss = agent.update([a.storage for a in agents])
                for k, v in loss.items():
                    if writer:
                        writer.add_scalar(f"agent{agent.agent_id}/{k}", v, j)
            adv_agents.train()

            for agent in agents:
                agent.storage.after_update()
            for i in range(len(agents)):
                adv_agents.storage[i].after_update()

            if j % log_interval == 0 and len(all_infos) > 1:
                squashed = _squash_info(all_infos)

                total_num_steps = (
                    (j + 1) * algorithm["num_processes"] * algorithm["num_steps"]
                )
                end = time.time()
                _log.info(
                    f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}"
                )
                _log.info(
                    f"Last {len(all_infos)} training episodes mean reward {squashed['episode_reward'].sum():.3f}"
                )

                for k, v in squashed.items():
                    _run.log_scalar(k, v, j)
                all_infos.clear()

            if save_interval is not None and (
                j > 0 and j % save_interval == 0 or j == num_updates
            ):
                cur_save_dir = path.join(save_dir, f"u{j}")
                for agent in agents:
                    save_at = path.join(cur_save_dir, f"agent{agent.agent_id}")
                    os.makedirs(save_at, exist_ok=True)
                    agent.save(save_at)
                save_at = path.join(cur_save_dir, f"adv_agents")
                os.makedirs(save_at, exist_ok=True)
                adv_agents.save(save_at)
                archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir, f"u{j}")
                shutil.rmtree(cur_save_dir)
                _run.add_artifact(archive_name)

            if eval_interval is not None and (
                j > 0 and j % eval_interval == 0 or j == num_updates
            ):
                evaluate(
                    agents, os.path.join(eval_dir, f"u{j}"),
                )
                videos = glob.glob(os.path.join(eval_dir, f"u{j}") + "/*.mp4")
                for i, v in enumerate(videos):
                    _run.add_artifact(v, f"u{j}.{i}.mp4")
    elif adv == "fgsm":
        # calculate the actions before purturbing
        clean_path = "/home/gwr/python_projects/Robust_Robotic_Warehouse/seac/results/unzip_models/rware-tiny-4ag-v1/u2000000" #"pretrained/rware-small-4ag"
        for agent in agents:
            agent.restore(clean_path + f"/agent{agent.agent_id}")
        with torch.no_grad():
            n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                            *[
                                agent.model.act(
                                    obs[agent.agent_id],
                                    agent.storage.recurrent_hidden_states[0],
                                    agent.storage.masks[0],
                                )
                                for agent in agents
                            ]
                        )
        adv_obs = obs

        for i in range(len(obs)):
            agents[i].storage.obs[0].copy_(adv_obs[i])
            agents[i].storage.to(algorithm["device"])

        start = time.time()
        num_updates = (
            int(num_env_steps) // algorithm["num_steps"] // algorithm["num_processes"]
        )

        all_infos = deque(maxlen=10)

        for j in range(1, num_updates + 1):

            for step in range(algorithm["num_steps"]):
                # Sample actions
                with torch.no_grad():
                    n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                        *[
                            agent.model.act(
                                agent.storage.obs[step],
                                agent.storage.recurrent_hidden_states[step],
                                agent.storage.masks[step],
                            )
                            for agent in agents
                        ]
                    )
                # Obser reward and next obs
                obs, reward, done, infos = envs.step(n_action) # n_action is the perturbed action
                with torch.no_grad():
                    _, temp_action, _, _ = zip(
                        *[
                            agent.model.act(
                                obs[agent.agent_id],
                                agent.storage.recurrent_hidden_states[step],
                                agent.storage.masks[step],
                            )
                            for agent in agents
                        ]
                    )
                adv_obs = fgsm(agents, epsilon_ball, obs, n_action, agents[0].optimizer)

                # envs.envs[0].render()

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

                bad_masks = torch.FloatTensor(
                    [
                        [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                        for info in infos
                    ]
                )
                for i in range(len(agents)):
                    agents[i].storage.insert(
                        adv_obs[i],
                        n_recurrent_hidden_states[i],
                        n_action[i],
                        n_action_log_prob[i],
                        n_value[i],
                        reward[:, i].unsqueeze(1),
                        masks,
                        bad_masks,
                    )

                for info in infos:
                    if info:
                        all_infos.append(info)

            # value_loss, action_loss, dist_entropy = agent.update(rollouts)
            for agent in agents:
                agent.compute_returns()
            for agent in agents:
                loss = agent.update([a.storage for a in agents])
                for k, v in loss.items():
                    if writer:
                        writer.add_scalar(f"agent{agent.agent_id}/{k}", v, j)
            for agent in agents:
                agent.storage.after_update()

            if j % log_interval == 0 and len(all_infos) > 1:
                squashed = _squash_info(all_infos)

                total_num_steps = (
                    (j + 1) * algorithm["num_processes"] * algorithm["num_steps"]
                )
                end = time.time()
                _log.info(
                    f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}"
                )
                _log.info(
                    f"Last {len(all_infos)} training episodes mean reward {squashed['episode_reward'].sum():.3f}"
                )

                for k, v in squashed.items():
                    _run.log_scalar(k, v, j)
                all_infos.clear()

            if save_interval is not None and (
                j > 0 and j % save_interval == 0 or j == num_updates
            ):
                cur_save_dir = path.join(save_dir, f"u{j}")
                for agent in agents:
                    save_at = path.join(cur_save_dir, f"agent{agent.agent_id}")
                    os.makedirs(save_at, exist_ok=True)
                    agent.save(save_at)
                archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir, f"u{j}")
                shutil.rmtree(cur_save_dir)
                _run.add_artifact(archive_name)

            if eval_interval is not None and (
                j > 0 and j % eval_interval == 0 or j == num_updates
            ):
                evaluate(
                    agents, os.path.join(eval_dir, f"u{j}"),
                )
                videos = glob.glob(os.path.join(eval_dir, f"u{j}") + "/*.mp4")
                for i, v in enumerate(videos):
                    _run.add_artifact(v, f"u{j}.{i}.mp4")
    elif adv == "pgd":
        # calculate the actions before purturbing
        clean_path = "/home/gwr/python_projects/Robust_Robotic_Warehouse/seac/results/unzip_models/rware-tiny-2ag-v1/u3000000" #"pretrained/rware-small-4ag"
        for agent in agents:
            agent.restore(clean_path + f"/agent{agent.agent_id}")
        with torch.no_grad():
            n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                            *[
                                agent.model.act(
                                    obs[agent.agent_id],
                                    agent.storage.recurrent_hidden_states[0],
                                    agent.storage.masks[0],
                                )
                                for agent in agents
                            ]
                        )
        adv_obs = obs

        for i in range(len(obs)):
            agents[i].storage.obs[0].copy_(adv_obs[i])
            agents[i].storage.to(algorithm["device"])

        start = time.time()
        num_updates = (
            int(num_env_steps) // algorithm["num_steps"] // algorithm["num_processes"]
        )

        all_infos = deque(maxlen=10)

        for j in range(1, num_updates + 1):

            for step in range(algorithm["num_steps"]):
                # Sample actions
                with torch.no_grad():
                    n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                        *[
                            agent.model.act(
                                agent.storage.obs[step],
                                agent.storage.recurrent_hidden_states[step],
                                agent.storage.masks[step],
                            )
                            for agent in agents
                        ]
                    )
                # Obser reward and next obs
                obs, reward, done, infos = envs.step(n_action) # n_action is the perturbed action
                with torch.no_grad():
                    _, temp_action, _, _ = zip(
                        *[
                            agent.model.act(
                                obs[agent.agent_id],
                                agent.storage.recurrent_hidden_states[step],
                                agent.storage.masks[step],
                            )
                            for agent in agents
                        ]
                    )
                adv_obs = pgd(agents, epsilon_ball, obs, n_action, agents[0].optimizer, 5)

                # envs.envs[0].render()

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

                bad_masks = torch.FloatTensor(
                    [
                        [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                        for info in infos
                    ]
                )
                for i in range(len(agents)):
                    agents[i].storage.insert(
                        adv_obs[i],
                        n_recurrent_hidden_states[i],
                        n_action[i],
                        n_action_log_prob[i],
                        n_value[i],
                        reward[:, i].unsqueeze(1),
                        masks,
                        bad_masks,
                    )

                for info in infos:
                    if info:
                        all_infos.append(info)

            # value_loss, action_loss, dist_entropy = agent.update(rollouts)
            for agent in agents:
                agent.compute_returns()
            for agent in agents:
                loss = agent.update([a.storage for a in agents])
                for k, v in loss.items():
                    if writer:
                        writer.add_scalar(f"agent{agent.agent_id}/{k}", v, j)
            for agent in agents:
                agent.storage.after_update()

            if j % log_interval == 0 and len(all_infos) > 1:
                squashed = _squash_info(all_infos)

                total_num_steps = (
                    (j + 1) * algorithm["num_processes"] * algorithm["num_steps"]
                )
                end = time.time()
                _log.info(
                    f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}"
                )
                _log.info(
                    f"Last {len(all_infos)} training episodes mean reward {squashed['episode_reward'].sum():.3f}"
                )

                for k, v in squashed.items():
                    _run.log_scalar(k, v, j)
                all_infos.clear()

            if save_interval is not None and (
                j > 0 and j % save_interval == 0 or j == num_updates
            ):
                cur_save_dir = path.join(save_dir, f"u{j}")
                for agent in agents:
                    save_at = path.join(cur_save_dir, f"agent{agent.agent_id}")
                    os.makedirs(save_at, exist_ok=True)
                    agent.save(save_at)
                archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir, f"u{j}")
                shutil.rmtree(cur_save_dir)
                _run.add_artifact(archive_name)

            if eval_interval is not None and (
                j > 0 and j % eval_interval == 0 or j == num_updates
            ):
                evaluate(
                    agents, os.path.join(eval_dir, f"u{j}"),
                )
                videos = glob.glob(os.path.join(eval_dir, f"u{j}") + "/*.mp4")
                for i, v in enumerate(videos):
                    _run.add_artifact(v, f"u{j}.{i}.mp4")
    elif adv == "rand_noise":
        adv_obs = rand_noise(epsilon_ball, obs)
        for i in range(len(obs)):
            agents[i].storage.obs[0].copy_(adv_obs[i])
            agents[i].storage.to(algorithm["device"])

        start = time.time()
        num_updates = (
            int(num_env_steps) // algorithm["num_steps"] // algorithm["num_processes"]
        )

        all_infos = deque(maxlen=10)

        for j in range(1, num_updates + 1):

            for step in range(algorithm["num_steps"]):
                # Sample actions
                with torch.no_grad():
                    n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                        *[
                            agent.model.act(
                                agent.storage.obs[step],
                                agent.storage.recurrent_hidden_states[step],
                                agent.storage.masks[step],
                            )
                            for agent in agents
                        ]
                    )
                # Obser reward and next obs
                obs, reward, done, infos = envs.step(n_action)
                # envs.envs[0].render()
                adv_obs = rand_noise(epsilon_ball, obs)

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

                bad_masks = torch.FloatTensor(
                    [
                        [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                        for info in infos
                    ]
                )
                for i in range(len(agents)):
                    agents[i].storage.insert(
                        adv_obs[i],
                        n_recurrent_hidden_states[i],
                        n_action[i],
                        n_action_log_prob[i],
                        n_value[i],
                        reward[:, i].unsqueeze(1),
                        masks,
                        bad_masks,
                    )

                for info in infos:
                    if info:
                        all_infos.append(info)

            # value_loss, action_loss, dist_entropy = agent.update(rollouts)
            for agent in agents:
                agent.compute_returns()

            for agent in agents:
                loss = agent.update([a.storage for a in agents])
                for k, v in loss.items():
                    if writer:
                        writer.add_scalar(f"agent{agent.agent_id}/{k}", v, j)

            for agent in agents:
                agent.storage.after_update()

            if j % log_interval == 0 and len(all_infos) > 1:
                squashed = _squash_info(all_infos)

                total_num_steps = (
                    (j + 1) * algorithm["num_processes"] * algorithm["num_steps"]
                )
                end = time.time()
                _log.info(
                    f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}"
                )
                _log.info(
                    f"Last {len(all_infos)} training episodes mean reward {squashed['episode_reward'].sum():.3f}"
                )

                for k, v in squashed.items():
                    _run.log_scalar(k, v, j)
                all_infos.clear()

            if save_interval is not None and (
                j > 0 and j % save_interval == 0 or j == num_updates
            ):
                cur_save_dir = path.join(save_dir, f"u{j}")
                for agent in agents:
                    save_at = path.join(cur_save_dir, f"agent{agent.agent_id}")
                    os.makedirs(save_at, exist_ok=True)
                    agent.save(save_at)
                archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir, f"u{j}")
                shutil.rmtree(cur_save_dir)
                _run.add_artifact(archive_name)

            if eval_interval is not None and (
                j > 0 and j % eval_interval == 0 or j == num_updates
            ):
                evaluate(
                    agents, os.path.join(eval_dir, f"u{j}"),
                )
                videos = glob.glob(os.path.join(eval_dir, f"u{j}") + "/*.mp4")
                for i, v in enumerate(videos):
                    _run.add_artifact(v, f"u{j}.{i}.mp4")
    elif adv == "gaussian_noise":
        adv_obs = gaussian_noise(epsilon_ball, obs)
        for i in range(len(obs)):
            agents[i].storage.obs[0].copy_(adv_obs[i])
            agents[i].storage.to(algorithm["device"])

        start = time.time()
        num_updates = (
            int(num_env_steps) // algorithm["num_steps"] // algorithm["num_processes"]
        )

        all_infos = deque(maxlen=10)

        for j in range(1, num_updates + 1):

            for step in range(algorithm["num_steps"]):
                # Sample actions
                with torch.no_grad():
                    n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                        *[
                            agent.model.act(
                                agent.storage.obs[step],
                                agent.storage.recurrent_hidden_states[step],
                                agent.storage.masks[step],
                            )
                            for agent in agents
                        ]
                    )
                # Obser reward and next obs
                obs, reward, done, infos = envs.step(n_action)
                # envs.envs[0].render()
                adv_obs = gaussian_noise(epsilon_ball, obs)

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

                bad_masks = torch.FloatTensor(
                    [
                        [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                        for info in infos
                    ]
                )
                for i in range(len(agents)):
                    agents[i].storage.insert(
                        adv_obs[i],
                        n_recurrent_hidden_states[i],
                        n_action[i],
                        n_action_log_prob[i],
                        n_value[i],
                        reward[:, i].unsqueeze(1),
                        masks,
                        bad_masks,
                    )

                for info in infos:
                    if info:
                        all_infos.append(info)

            # value_loss, action_loss, dist_entropy = agent.update(rollouts)
            for agent in agents:
                agent.compute_returns()

            for agent in agents:
                loss = agent.update([a.storage for a in agents])
                for k, v in loss.items():
                    if writer:
                        writer.add_scalar(f"agent{agent.agent_id}/{k}", v, j)

            for agent in agents:
                agent.storage.after_update()

            if j % log_interval == 0 and len(all_infos) > 1:
                squashed = _squash_info(all_infos)

                total_num_steps = (
                    (j + 1) * algorithm["num_processes"] * algorithm["num_steps"]
                )
                end = time.time()
                _log.info(
                    f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}"
                )
                _log.info(
                    f"Last {len(all_infos)} training episodes mean reward {squashed['episode_reward'].sum():.3f}"
                )

                for k, v in squashed.items():
                    _run.log_scalar(k, v, j)
                all_infos.clear()

            if save_interval is not None and (
                j > 0 and j % save_interval == 0 or j == num_updates
            ):
                cur_save_dir = path.join(save_dir, f"u{j}")
                for agent in agents:
                    save_at = path.join(cur_save_dir, f"agent{agent.agent_id}")
                    os.makedirs(save_at, exist_ok=True)
                    agent.save(save_at)
                archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir, f"u{j}")
                shutil.rmtree(cur_save_dir)
                _run.add_artifact(archive_name)

            if eval_interval is not None and (
                j > 0 and j % eval_interval == 0 or j == num_updates
            ):
                evaluate(
                    agents, os.path.join(eval_dir, f"u{j}"),
                )
                videos = glob.glob(os.path.join(eval_dir, f"u{j}") + "/*.mp4")
                for i, v in enumerate(videos):
                    _run.add_artifact(v, f"u{j}.{i}.mp4")
    else:
        clean_path = "/home/gwr/python_projects/Robust_Robotic_Warehouse/seac/results/unzip_models/rware-tiny-4ag-v1/u2000000" #"pretrained/rware-small-4ag"
        for agent in agents:
            agent.restore(clean_path + f"/agent{agent.agent_id}")
        for i in range(len(obs)):
            agents[i].storage.obs[0].copy_(obs[i])
            agents[i].storage.to(algorithm["device"])

        start = time.time()
        num_updates = (
            int(num_env_steps) // algorithm["num_steps"] // algorithm["num_processes"]
        )

        all_infos = deque(maxlen=10)

        for j in range(1, num_updates + 1):

            for step in range(algorithm["num_steps"]):
                # Sample actions
                with torch.no_grad():
                    n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                        *[
                            agent.model.act(
                                agent.storage.obs[step],
                                agent.storage.recurrent_hidden_states[step],
                                agent.storage.masks[step],
                            )
                            for agent in agents
                        ]
                    )
                # Obser reward and next obs
                obs, reward, done, infos = envs.step(n_action)
                # envs.envs[0].render()

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

                bad_masks = torch.FloatTensor(
                    [
                        [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                        for info in infos
                    ]
                )
                for i in range(len(agents)):
                    agents[i].storage.insert(
                        obs[i],
                        n_recurrent_hidden_states[i],
                        n_action[i],
                        n_action_log_prob[i],
                        n_value[i],
                        reward[:, i].unsqueeze(1),
                        masks,
                        bad_masks,
                    )

                for info in infos:
                    if info:
                        all_infos.append(info)

            # value_loss, action_loss, dist_entropy = agent.update(rollouts)
            for agent in agents:
                agent.compute_returns()

            for agent in agents:
                loss = agent.update([a.storage for a in agents])
                for k, v in loss.items():
                    if writer:
                        writer.add_scalar(f"agent{agent.agent_id}/{k}", v, j)

            for agent in agents:
                agent.storage.after_update()

            if j % log_interval == 0 and len(all_infos) > 1:
                squashed = _squash_info(all_infos)

                total_num_steps = (
                    (j + 1) * algorithm["num_processes"] * algorithm["num_steps"]
                )
                end = time.time()
                _log.info(
                    f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}"
                )
                _log.info(
                    f"Last {len(all_infos)} training episodes mean reward {squashed['episode_reward'].sum():.3f}"
                )

                for k, v in squashed.items():
                    _run.log_scalar(k, v, j)
                all_infos.clear()

            if save_interval is not None and (
                j > 0 and j % save_interval == 0 or j == num_updates
            ):
                cur_save_dir = path.join(save_dir, f"u{j}")
                for agent in agents:
                    save_at = path.join(cur_save_dir, f"agent{agent.agent_id}")
                    os.makedirs(save_at, exist_ok=True)
                    agent.save(save_at)
                archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir, f"u{j}")
                shutil.rmtree(cur_save_dir)
                _run.add_artifact(archive_name)

            if eval_interval is not None and (
                j > 0 and j % eval_interval == 0 or j == num_updates
            ):
                evaluate(
                    agents, os.path.join(eval_dir, f"u{j}"),
                )
                videos = glob.glob(os.path.join(eval_dir, f"u{j}") + "/*.mp4")
                for i, v in enumerate(videos):
                    _run.add_artifact(v, f"u{j}.{i}.mp4")
    envs.close()
