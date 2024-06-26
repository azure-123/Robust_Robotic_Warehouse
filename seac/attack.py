import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

def tar_attack(agents, epsilon, states, action, tar_action, opt):
    loss_func = nn.CrossEntropyLoss()
    adv_states = [Variable(adv_state, requires_grad=True) for adv_state in states]
    logits = [agent.model.attack_act(
                                adv_states[agent.agent_id], 
                                agent.storage.recurrent_hidden_states,
                                agent.storage.masks
                            ) 
                            for agent in agents]
    for adv_state in adv_states:
        adv_state.retain_grad()
    opt.zero_grad()
    losses = [-loss_func(logits[i], tar_action[i].squeeze()) + loss_func(logits[i], action[i].squeeze()) for i in range(len(agents))]
    for loss in losses:
        loss.backward()
    
    eta = [epsilon * adv_states[i].grad.data.sign() for i in range(len(agents))]
    for i in range(len(adv_states)):
        adv_states[i].data = Variable(adv_states[i].data + eta[i], requires_grad=True)
        eta[i] = torch.clamp(adv_states[i].data - states[i].data, -epsilon, epsilon)
        adv_states[i].data = states[i].data + eta[i]
        adv_states[i] = adv_states[i].detach()

    return adv_states

# def tar_attack_eval(model, epsilon, states, action, tar_action, opt, use_cuda):
    # loss_func = nn.CrossEntropyLoss()
    # adv_states = torch.tensor(np.array(states), requires_grad=True, dtype=torch.float32)
    # if use_cuda:
    #     adv_states = adv_states.cuda()
    # logits = model(adv_states)

    # if use_cuda:
    #     loss = -loss_func(logits, torch.tensor(np.array(tar_action)).long().cuda()) + loss_func(logits, torch.tensor(np.array(action)).long().cuda())
    # else:
    #     loss = -loss_func(logits, torch.tensor(np.array(tar_action)).long()) + loss_func(logits, torch.tensor(np.array(action)).long())
    
    # adv_states.retain_grad()
    # opt.zero_grad()
    # loss.backward()
    

    # eta_0 = epsilon * adv_states.grad.data.sign()
    # adv_states.data = Variable(adv_states.data + eta_0, requires_grad=True)

    # if use_cuda:
    #     eta_0 = torch.clamp(adv_states.data - torch.tensor(np.array(states)).cuda().data, -epsilon, epsilon)
    #     adv_states.data = torch.tensor(np.array(states)).cuda().data + eta_0
    # else:
    #     eta_0 = torch.clamp(adv_states.data - torch.tensor(np.array(states)).data, -epsilon, epsilon)
    #     adv_states.data = torch.tensor(np.array(states)).data + eta_0

    # return adv_states.cpu().data.numpy()

def atla(adv_agents, epsilon, states):
    process_obs = torch.stack(states).transpose(0, 1)
    mean, std = adv_agents.actor(process_obs)
    perturbations = F.hardtanh(torch.distributions.Normal(mean, std).sample()) * epsilon
                     # perturbation generated by the adversary
    adv_states = process_obs + perturbations # add the perturbation to the states
    adv_states = [adv_state for adv_state in adv_states.transpose(0, 1)]
    perturbations = [perturbation for perturbation in perturbations.transpose(0, 1)]
    return adv_states, perturbations

def fgsm(agents, epsilon, states, action, opt):
    loss_func = nn.CrossEntropyLoss()
    adv_states = [Variable(adv_state, requires_grad=True) for adv_state in states]
    logits = [agent.model.attack_act(
                                adv_states[agent.agent_id], 
                                agent.storage.recurrent_hidden_states,
                                agent.storage.masks
                            ) 
                            for agent in agents]
    for adv_state in adv_states:
        adv_state.retain_grad()
    opt.zero_grad()
    losses = [loss_func(logits[i], action[i].squeeze()) for i in range(len(agents))]
    for loss in losses:
        loss.backward()
    eta = [epsilon * adv_states[i].grad.data.sign() for i in range(len(agents))]
    for i in range(len(adv_states)):
        adv_states[i].data = Variable(adv_states[i].data + eta[i], requires_grad=True)
        eta[i] = torch.clamp(adv_states[i].data - states[i].data, -epsilon, epsilon)
        adv_states[i].data = states[i].data + eta[i]
        adv_states[i] = adv_states[i].detach()
    return adv_states

def pgd(agents, epsilon, states, action, opt, niters):
    loss_func = nn.CrossEntropyLoss()
    adv_states = [Variable(adv_state, requires_grad=True) for adv_state in states]

    noise = [2 * epsilon * torch.rand(adv_state.size()) - epsilon for adv_state in adv_states]
    for i in range(len(adv_states)):
        adv_states[i].data = Variable(adv_states[i].data + noise[i], requires_grad=True)
        noise[i] = torch.clamp(adv_states[i].data - states[i].data, -epsilon, epsilon)
        adv_states[i].data = states[i].data + noise[i]
        adv_states[i] = Variable(adv_states[i], requires_grad=True)

    # iterate
    step_size = epsilon / niters
    for _ in range(niters):
        logits = [agent.model.attack_act(
                                adv_states[agent.agent_id], 
                                agent.storage.recurrent_hidden_states,
                                agent.storage.masks
                            ) 
                            for agent in agents]
        opt.zero_grad()
        losses = [loss_func(logits[i], action[i].squeeze()) for i in range(len(agents))]
        for adv_state in adv_states:
            adv_state.retain_grad()
        for loss in losses:
            loss.backward()
        eta = [step_size * adv_states[i].grad.data.sign() for i in range(len(agents))]
        for i in range(len(adv_states)):
            adv_states[i].data = Variable(adv_states[i].data + eta[i], requires_grad=True)
            eta[i] = torch.clamp(adv_states[i].data - states[i].data, -epsilon, epsilon)
            adv_states[i].data = states[i].data + eta[i]
    for i in range(len(adv_states)):
        adv_states[i] = adv_states[i].detach()
    return adv_states

def rand_noise(epsilon, states):
    adv_states = [Variable(adv_state) for adv_state in states]
    eta = [2 * epsilon * torch.rand(adv_state.size()) - epsilon for adv_state in adv_states]
    for i in range(len(adv_states)):
        adv_states[i].data = Variable(adv_states[i].data + eta[i], requires_grad=True)
        eta[i] = torch.clamp(adv_states[i].data - states[i].data, -epsilon, epsilon)
        adv_states[i].data = states[i].data + eta[i]
        adv_states[i] = adv_states[i].detach()
    # print(X_adv - agent_inputs)
    return adv_states

def gaussian_noise(epsilon, states):
    adv_states = [Variable(adv_state) for adv_state in states]
    eta = [2 * epsilon * torch.randn(adv_state.size()) - epsilon for adv_state in adv_states]
    for i in range(len(adv_states)):
        adv_states[i].data = Variable(adv_states[i].data + eta[i], requires_grad=True)
        eta[i] = torch.clamp(adv_states[i].data - states[i].data, -epsilon, epsilon)
        adv_states[i].data = states[i].data + eta[i]
        adv_states[i] = adv_states[i].detach()
    # print(X_adv - agent_inputs)
    return adv_states